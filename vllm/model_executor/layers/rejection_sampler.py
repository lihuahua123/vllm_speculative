# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from importlib.util import find_spec
from typing import Dict, Optional, Tuple

import torch
import torch.jit

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeStochasticBaseSampler)
from vllm.platforms import current_platform
import pickle
import os
logger = init_logger(__name__)

if find_spec("flashinfer"):
    """
    Consider utilizing the FlashInfer rejection sampling kernel initially,
    as it employs a dedicated kernel rather than relying on 
    Torch tensor operations. This design choice helps to fuse operations, 
    reduce memory I/O, and consequently enhances performance.
    """
    from flashinfer.sampling import chain_speculative_sampling
else:
    chain_speculative_sampling = None


class RejectionSampler(SpecDecodeStochasticBaseSampler):
    """Apply modified rejection sampling as described in "Accelerating Large
        Language Model Decoding with Speculative Sampling"
        https://arxiv.org/pdf/2302.01318.pdf.
    """

    def __init__(self,
                 strict_mode: bool = False,
                 use_flashinfer: Optional[bool] = None):
        """Create a rejection sampler.

        Args:
            strict_mode: Whether or not to perform shape/device/dtype checks
            during sampling. This catches correctness issues but adds
            nontrivial latency.
            use_flashinfer: We will use this parameter to determine whether
            to use the FlashInfer rejection sampling kernel or not. If it's
            None, we will use the default value from the environment variable.
            This parameter is only used for testing purposes.
        """
        super().__init__(strict_mode=strict_mode)
        if use_flashinfer is None:
            self.use_flashinfer = envs.VLLM_USE_FLASHINFER_SAMPLER and (
                chain_speculative_sampling is not None)
        else:
            self.use_flashinfer = use_flashinfer

        if self.use_flashinfer:
            logger.info("Use flashinfer for rejection sampling.")
        else:
            logger.info("Use pytorch for rejection sampling.")
            
        self.selected_target_probs = []
        self.selected_draft_probs = []
        self.target_probs = []
        self.draft_probs = []

    def forward(
        self,
        target_with_bonus_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        seeded_seqs: Optional[Dict[int, torch.Generator]] = None,
    ) -> torch.Tensor:
        """Sample token ids using rejection sampling. This accepts or rejects
        tokens proposed by the draft model using the probability of each token
        according to the draft and target models.

        In the worst case where all draft tokens are rejected, it is guaranteed
        one correct token will be emitted.

        In the case where all draft tokens are accepted, a bonus token will be
        accepted as its cheap to have the target model score this speculative
        sequence.

        Args:
            target_with_bonus_probs: The probability distribution 
                over token ids given context according to the target model.
            shape = [batch_size, num_speculative_tokens + 1, vocab_size]

            bonus_token_ids: The "bonus" token ids that are accepted iff all
                speculative tokens in a sequence are accepted.
            shape = [batch_size, num_bonus_tokens]

            draft_probs: The probability distribution over token ids given
                context according to the draft model.
            shape = [batch_size, num_speculative_tokens, vocab_size]

            draft_token_ids: The token ids that were sampled from the draft
                probabilities.
            shape = [batch_size, num_speculative_tokens]

            seeded_seqs: Dict of batch row index to torch generator, for
                sequences using seeded generation.

        Returns:
            output_token_ids: The token ids sampled via rejection sampling,
                or -1 if unable to sample a token because the previous token
                was rejected.
            shape = [batch_size, num_speculative_tokens + num_bonus_tokens]
        """
        # Only perform shape/dtype/device checking in strict mode, as it adds
        # overhead.
        if self._strict_mode:
            self._raise_if_incorrect_input(target_with_bonus_probs,
                                           draft_token_ids, bonus_token_ids,
                                           draft_probs)

        batch_size, k, _ = draft_probs.shape

        # batch_size = 0 when all requests in the batch are
        # non_spec requests. In this case, output_token_ids is
        # just an empty tensor.
        if batch_size == 0:
            return torch.empty(0, k + 1, device=draft_probs.device, dtype=int)

        # If use Flashinfer chain_speculative_sampling kernel
        # for rejection sampling
        if self.use_flashinfer and chain_speculative_sampling is not None:
            batch_size, k, _ = draft_probs.shape
            uniform_samples = self._create_uniform_samples(
                seeded_seqs, batch_size, k, draft_probs.device)
            output_token_ids, accepted_token_num, emitted_token_num \
                = chain_speculative_sampling(
                draft_probs, draft_token_ids, uniform_samples,
                target_with_bonus_probs)

            # num_emitted_tokens returned by flashinfer
            # does not include the bonus token
            # Flashinfer stops at the first token that violates
            # the condition p >= q and does not include recovery/bonus token.
            # Therefore, we need to add batch_size here.
            self.num_accepted_tokens += accepted_token_num.sum()
            self.num_emitted_tokens += emitted_token_num.sum() + batch_size
            self.num_draft_tokens += batch_size * k
            self.ratio = accepted_token_num.sum() / (batch_size * k)
        else:
            accepted, recovered_token_ids = (
                self._batch_modified_rejection_sampling(
                    target_with_bonus_probs[:, :-1],
                    draft_probs,
                    draft_token_ids,
                    seeded_seqs,
                ))

            output_token_ids = self._create_output(
                accepted,
                recovered_token_ids,
                draft_token_ids,
                bonus_token_ids,
            )
        return output_token_ids

    def _batch_modified_rejection_sampling(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform modified rejection sampling on each sequence.

        Returns:
            A tuple of two tensors:
            0: A bool tensor of which tokens in each sequence is accepted.
                shape = [batch_size, k]
            1: Token ids sampled from a recovered distribution, to be used
                when a token is rejected.
                shape = [batch_size, k]
        """

        batch_size, k, vocab_size = draft_probs.shape

        # shape [batch_size, k]
        accepted = self._get_accepted(target_probs, draft_probs,
                                      draft_token_ids, seeded_seqs)

        recovered_probs = self._get_recovered_probs(
            target_probs, draft_probs).reshape(batch_size * k, vocab_size)

        # NOTE: the recovered_probs are overwritten by this method.
        recovered_token_ids = _multinomial(
            recovered_probs,
            num_samples=1,
            k=k,
            seeded_seqs=seeded_seqs or {},
        ).reshape(batch_size, k)

        return accepted, recovered_token_ids

    def _create_uniform_samples(self,
                                seeded_seqs: Optional[Dict[int,
                                                           torch.Generator]],
                                batch_size: int, k: int,
                                device: torch.device) -> torch.Tensor:
        """
        Generates a batch of uniform random samples, with optional seeding 
        for specific sequences.

        This method creates a tensor of shape `(batch_size, k + 1)` filled 
        with uniform random values in the range [0, 1). If `seeded_seqs` 
        is provided, the sequences corresponding to specific indices 
        will be generated using the provided `torch.Generator` for 
        reproducibility. The other sequences will be generated without 
        a seed.

        Args:
            seeded_seqs : Optional[Dict[int, torch.Generator]]
                A dictionary mapping indices in the batch to 
                `torch.Generator` objects. If `None`, all samples are 
                generated without a seed.
            batch_size : int
                The number of sequences to generate.
            k : int
                The number of random samples per sequence.
            device : torch.device
                The device on which to allocate the tensor.

        Returns:
            uniform_rand : torch.Tensor
                A tensor of shape `(batch_size, k + 1)` containing uniform 
                random values in the range [0, 1).
        """
        if not seeded_seqs:
            return torch.rand(batch_size, k + 1, device=device)

        uniform_rand = torch.empty(batch_size, k + 1, device=device)

        non_seeded_indices = []
        for idx in range(batch_size):
            generator = seeded_seqs.get(idx)
            if generator is None:
                non_seeded_indices.append(idx)
            else:
                uniform_rand[idx, :] = torch.rand(1,
                                                  k + 1,
                                                  dtype=self.probs_dtype,
                                                  device=device,
                                                  generator=generator)
        if non_seeded_indices:
            uniform_rand[non_seeded_indices, :] = torch.rand(
                len(non_seeded_indices),
                k + 1,
                dtype=self.probs_dtype,
                device=device)
        return uniform_rand


    def _get_accepted2(
        self,
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> torch.Tensor: 
        # 降低接收阈值
        # Alignment-Augmented Speculative Decoding with Alignment Sampling and Conditional Verification
        # 这个好像没有什么效果
        batch_size, k, _ = draft_probs.shape
        device = target_probs.device
        
        # 创建索引
        batch_indices = torch.arange(batch_size, device=device)[:, None]
        prob_indices = torch.arange(k, device=device)
        
        # 计算信息熵和自适应阈值
        epsilon = 1e-10
        alpha = 0.1  # hyperparameter from paper (0.1 for LLaMA3)
        beta = 0.1   # hyperparameter from paper
        entropy = -torch.sum(target_probs * torch.log(target_probs + epsilon), dim=-1)
        max_probs = torch.max(target_probs, dim=-1).values
        adaptive_threshold = torch.minimum(-alpha * entropy + beta, max_probs)
        
        # 获取目标概率和草稿概率
        selected_target_probs = target_probs[batch_indices, prob_indices, draft_token_ids]
        selected_draft_probs = draft_probs[batch_indices, prob_indices, draft_token_ids]
        
        # 生成随机数
        # uniform_rand = self._create_uniform_samples(seeded_seqs, batch_size, k, device)
        uniform_rand = self._create_uniform_samples(seeded_seqs, batch_size,
                                                    k - 1, target_probs.device)
        # 确保阈值张量形状正确 [batch_size, k]
        if adaptive_threshold.dim() == 1:
            adaptive_threshold = adaptive_threshold.unsqueeze(1).expand(-1, k)
        
        # 计算接受条件
        above_threshold = selected_target_probs >= adaptive_threshold
        # capped_ratio = torch.minimum(
        #     selected_target_probs / selected_draft_probs,
        #     torch.ones_like(selected_target_probs))
        capped_ratio = torch.minimum(
            selected_target_probs / selected_draft_probs,
            torch.full((1, ), 1, device=target_probs.device))
        accepted = (uniform_rand < capped_ratio) & above_threshold
        
        return accepted
    
    def _get_accepted(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> torch.Tensor:
        r"""Create bool matrix over the proposed draft tokens. If
        True, then a token can be accepted, else it should be
        rejected.

        Given :math:`q(\hat{x}_{n+1}|x_1, \dots, x_n)`, the probability of
        :math:`\hat{x}_{n+1}` given context :math:`x_1, \dots, x_n` according
        to the target model, and :math:`p(\hat{x}_{n+1}|x_1, \dots, x_n)`, the
        same conditional probability according to the draft model, the token
        is accepted with probability:

        .. math::
            \min\left(1, \frac{q(\hat{x}_{n+1}|x_1, \dots, x_n)}
                           {p(\hat{x}_{n+1}|x_1, \dots, x_n)}\right)

        This implementation does not apply causality. When using the output,
        if a token is rejected, subsequent tokens should not be used.

        Returns a bool tensor of shape [batch_size, k] specifying which tokens
        are accepted.
        """
        batch_size, k, _ = draft_probs.shape
        batch_indices = torch.arange(batch_size,
                                     device=target_probs.device)[:, None]
        probs_indicies = torch.arange(k, device=target_probs.device)

        # Calculate posterior entropy (similar to TypicalAcceptanceSampler)
        # epsilon = 1e-5
        # posterior_entropy = -torch.sum(
        #     target_probs * torch.log(target_probs + epsilon), dim=-1)
        
        # # Calculate maximum entropy and normalized entropy
        # vocab_size = target_probs.shape[-1]
        # max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float, device=target_probs.device))
        # normalized_entropy = posterior_entropy / max_entropy
        
        # # Print entropy information
        # print("posterior_entropy.shape", posterior_entropy.shape)
        # print("max_entropy:", max_entropy.item())
        # print("normalized_entropy:", normalized_entropy)

        # shape [batch_size, k]
        selected_draft_probs = draft_probs[batch_indices, probs_indicies,
                                           draft_token_ids]

        # shape [batch_size, k]
        selected_target_probs = target_probs[batch_indices, probs_indicies,
                                             draft_token_ids]
        self.target_probs.append(target_probs)
        self.draft_probs.append(draft_probs)
        self.selected_target_probs.append(selected_target_probs)
        self.selected_draft_probs.append(selected_draft_probs)
        # print("selected_target_probs",selected_target_probs)
        # print("selected_draft_probs",selected_draft_probs)
        uniform_rand = self._create_uniform_samples(seeded_seqs, batch_size,
                                                    k - 1, target_probs.device)
        capped_ratio = torch.minimum(
            selected_target_probs / selected_draft_probs,
            torch.full((1, ), 1, device=target_probs.device))
        # accepted = torch.zeros_like(capped_ratio, dtype=torch.bool)
        accepted = uniform_rand < capped_ratio
        return accepted
    
    
    def _get_ada_edl_lower_bound(
        self,
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        gamma: float = 0.5,  # hyperparameter γ
    ) -> torch.Tensor:
        """Calculate AdaEDL entropy-based lower bound on token acceptance probability.
        
        AdaEDL formula: 1 - √(γ * HDM(x))
        where HDM(x) is the entropy of the draft model probability distribution.
        
        Args:
            draft_probs: Draft model probability distribution
            gamma: Hyperparameter γ for controlling the bound tightness
            
        Returns:
            lower_bound: Shape [batch_size, k], lower bound on acceptance rate
        """
        epsilon = 1e-10
        
        # Calculate entropy HDM(x) = -Σ(pDM(x) * log(pDM(x)))
        # Shape: [batch_size, k]
        draft_entropy = -torch.sum(
            draft_probs * torch.log(draft_probs + epsilon), dim=-1)
        
        # Apply AdaEDL formula: 1 - √(γ * HDM(x))
        # Clamp to ensure we don't take sqrt of negative values
        gamma_entropy = torch.clamp(gamma * draft_entropy, min=0.0)
        lower_bound = 1.0 - torch.sqrt(gamma_entropy)
        
        # Ensure lower bound is in valid range [0, 1]
        lower_bound = torch.clamp(lower_bound, min=0.0, max=1.0)
        
        return lower_bound

    def _get_accepted_with_ada_edl(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]],
        use_ada_edl: bool = False,
        gamma: float = 0.5,
    ) -> torch.Tensor:
        """Enhanced acceptance function with AdaEDL entropy-based stopping criteria.
        
        Args:
            target_probs: Target model probabilities
            draft_probs: Draft model probabilities  
            draft_token_ids: Sampled draft tokens
            seeded_seqs: Seeded generators for reproducibility
            use_ada_edl: Whether to use AdaEDL entropy-based bound
            gamma: AdaEDL hyperparameter γ
            
        Returns:
            accepted: Boolean tensor indicating which tokens are accepted
        """
        batch_size, k, _ = draft_probs.shape
        batch_indices = torch.arange(batch_size,
                                     device=target_probs.device)[:, None]
        probs_indicies = torch.arange(k, device=target_probs.device)

        # shape [batch_size, k]
        selected_draft_probs = draft_probs[batch_indices, probs_indicies,
                                           draft_token_ids]
        selected_target_probs = target_probs[batch_indices, probs_indicies,
                                             draft_token_ids]
        
        # Store for analysis
        self.selected_target_probs.append(selected_target_probs)
        self.selected_draft_probs.append(selected_draft_probs)
        
        
        # Calculate AdaEDL entropy-based lower bound
        ada_edl_bound = self._get_ada_edl_lower_bound(draft_probs, gamma)
        
        # Apply AdaEDL stopping criteria
        # Only accept if the acceptance probability is above the lower bound
        acceptance_prob = torch.minimum(
            selected_target_probs / selected_draft_probs,
            torch.ones_like(selected_target_probs))
        
       
        # Combine standard rejection sampling with AdaEDL criteria
        accepted = ada_edl_bound >= 0.5 #ada_edl_condition
        
        return accepted
    
    def save_selected_probs(self):
        output_dir = "confidence_history"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "selected_target_probs.pkl"), "wb") as f:
            pickle.dump(self.selected_target_probs, f)
        with open(os.path.join(output_dir, "selected_draft_probs.pkl"), "wb") as f:
            pickle.dump(self.selected_draft_probs, f)
        with open(os.path.join(output_dir, "target_probs.pkl"), "wb") as f:
            pickle.dump(self.target_probs, f)
        with open(os.path.join(output_dir, "draft_probs.pkl"), "wb") as f:
            pickle.dump(self.draft_probs, f)
            
    def _get_recovered_probs(
            self,
            target_probs: torch.Tensor,  # [k, vocab_size]
            draft_probs: torch.Tensor,  # [k, vocab_size]
    ) -> torch.Tensor:
        r"""Create a probability distribution for each proposed token which can
        be sampled if the proposed token is rejected.

        When this routine is applied sequentially, the true distribution of the
        target model is recovered (within hardware numerics).

        The probability distribution used in this rejection case is constructed
        as follows. Given :math:`q(x|x_1, \dots, x_n)`, the probability of
        :math:`x` given context :math:`x_1, \dots, x_n` according to the target
        model and :math:`p(x|x_1, \dots, x_n)`, the same conditional probability
        according to the draft model:

        .. math::
            x_{n+1} \sim (q(x|x_1, \dots, x_n) - p(x|x_1, \dots, x_n))_+

        where :math:`(f(x))_+` is defined as:

        .. math::
            (f(x))_+ = \frac{\max(0, f(x))}{\sum_x \max(0, f(x))}

        See https://github.com/vllm-project/vllm/pull/2336 for a visualization
        of the draft, target, and recovered probability distributions.

        Returns a tensor of shape [batch_size, k, vocab_size].

        Note: This batches operations on GPU and thus constructs the recovered
        distribution for all tokens, even if they are accepted. This causes
        division-by-zero errors, so we use self._smallest_positive_value to
        avoid that. This introduces some drift to the distribution.
        """
        _, k, _ = draft_probs.shape

        # shape [batch_size, k, vocab_size]
        difference = target_probs - draft_probs

        # TODO(cade): Can we use logprobs instead of probs, and avoid the
        # division-by-zero errors without introducing distribution drift?

        # shape [batch_size, k, vocab_size]
        f = torch.clamp(difference, min=self._smallest_positive_value)

        # shape [batch_size, k, vocab_size]
        recovered_probs = f / torch.sum(f, dim=-1).reshape(-1, k, 1)

        return recovered_probs

    @cached_property
    def _smallest_positive_value(self) -> float:
        """Return the smallest positive value representable by the probs dtype.
        This value is used when constructing a distribution from which to sample
        recovered tokens in the first rejection case.

        See _get_recovered_probs for more details

        Note that this isn't actually the smallest positive value representable
        by float32, but the smallest positive normal value.
        See https://en.wikipedia.org/wiki/Subnormal_number for more information.
        """
        return torch.finfo(self.probs_dtype).tiny


# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead that skips the sync.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    k: int,
    seeded_seqs: Dict[int, torch.Generator],
) -> torch.Tensor:

    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
                                             -1, probs.shape[1])
    q = torch.empty_like(probs)
    if not seeded_seqs:
        q.exponential_(1.0)
    else:
        start = 0
        for idx in range(len(q) // k):
            end = start + k
            generator = seeded_seqs.get(idx)
            # Note: generator might be None for non seeded
            q[start:end].exponential_(1.0, generator=generator)
            start = end

    return probs.div_(q).argmax(dim=1).view(-1, num_samples)
