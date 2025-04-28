from vllm.entrypoints.api_server import FlexibleArgumentParser
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio
import ssl
from vllm.engine.llm_engine_with_ilp import LLMEngineWithILP
from vllm.entrypoints.api_server import run_server
from vllm.engine.llm_engine import UsageContext
from vllm.engine.async_llm_engine import AsyncLLMEngine
if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=parser.check_port, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--enable-ssl-refresh",
        action="store_true",
        default=False,
        help="Refresh SSL Context when SSL certificate files change")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    parser.add_argument("--strategy", type=str, default="baseline", 
                        choices=["baseline", "ilp", "no-spec"],
                        help="策略名称")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    asyncio.run(run_server(args, llm_engine=AsyncLLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.API_SERVER)))
    # if args.strategy != "ilp":
    #     asyncio.run(run_server(args, llm_engine=AsyncLLMEngine.from_engine_args(
    #               engine_args, usage_context=UsageContext.API_SERVER)))
    # elif args.strategy == "ilp":
    #     asyncio.run(run_server(args, llm_engine=LLMEngineWithILP.from_engine_args(
    #               engine_args, usage_context=UsageContext.API_SERVER)))
