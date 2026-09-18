import logging
logging.basicConfig(level=logging.INFO, force=True)

from settings import Settings
from evaluator import Evaluator
from runner import Runner

settings = Settings()
evaluator = Evaluator(settings)
runner = Runner(settings)

questions = evaluator.get_questions()
print(f"\nRunning all {len(questions)} questions...\n")

df = runner.run_agent(questions, username=settings.username)
print("\n--- RESULTS ---")
print(df.to_string())