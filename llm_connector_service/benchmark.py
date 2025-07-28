import time
import sys
import logging
from retrieval import sap_retrieval, SAPDefinition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <prompt>")
        sys.exit(1)
    prompt = " ".join(sys.argv[1:])
    logger.info(f"Benchmarking retrieval for prompt: {prompt}")
    start = time.time()
    results = sap_retrieval.query(prompt, top_k=3)
    elapsed = time.time() - start
    print(f"Retrieval time: {elapsed:.4f} seconds\n")
    print("Top-3 matched SAP definitions:")
    for i, (definition, score) in enumerate(results, 1):
        print(f"{i}. {definition} (Distance: {score:.4f})")

if __name__ == "__main__":
    main() 