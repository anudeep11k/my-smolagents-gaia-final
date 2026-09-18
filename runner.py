import re
from settings import Settings
from models import Question, QuestionAnswerPair, GoogleModelID
from agent import ManagerAgent
from tools import GetTaskFileTool, VideoUnderstandingTool, AudioUnderstandingTool
import pandas as pd
import logging
import json
import asyncio
import nest_asyncio
nest_asyncio.apply()
logger = logging.getLogger(__name__)

YOUTUBE_URL_PATTERN = re.compile(r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://youtu\.be/[\w-]+")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg")


class Runner():
    def __init__(self, settings: Settings):
        self.settings = settings
        self.file_tool = GetTaskFileTool(settings)
        self.video_tool = VideoUnderstandingTool(settings, GoogleModelID.GEMINI_2_0_FLASH)
        self.audio_tool = AudioUnderstandingTool(settings, GoogleModelID.GEMINI_2_0_FLASH)

    def _save_pairs(self, pairs: list[QuestionAnswerPair], username: str):
        """Write the question answer pairs to a user-specific file."""
        answers = [pair.model_dump() for pair in pairs if pair is not None]
        file_name = f"answers_{username}.json"
        with open(file_name, "w") as f:
            json.dump(answers, f, indent=4)

    def _prefetch_video_description(self, question_text: str) -> str:
        """If the question references a YouTube video, describe it now and return the description."""
        match = YOUTUBE_URL_PATTERN.search(question_text)
        if not match:
            return ""
        youtube_url = match.group(0)
        try:
            description = self.video_tool.forward(
                youtube_url=youtube_url,
                prompt=(
                    "Describe this video in detail: any spoken dialogue or captions, "
                    "any visually countable elements (people, animals, objects on screen), "
                    "and anything else relevant to answering questions about its content."
                ),
            )
            return f"\n\nPre-fetched video description for {youtube_url}: {description}"
        except Exception as e:
            logger.error(f"Video pre-fetch failed for {youtube_url}: {e}")
            return f"\n\nNote: automatic video description failed for {youtube_url}: {e}"

    def _prefetch_audio_transcript(self, task_id: str, file_name: str) -> str:
        """If the file is an audio file, download and transcribe/describe it now."""
        if not file_name or not file_name.lower().endswith(AUDIO_EXTENSIONS):
            return ""
        try:
            file_path = self.file_tool.forward(task_id=task_id, file_name=file_name)
            transcript = self.audio_tool.forward(
                file_path=file_path,
                prompt="Transcribe this audio in full detail, word for word if possible.",
            )
            return f"\n\nPre-fetched audio transcript for {file_name}: {transcript}"
        except Exception as e:
            logger.error(f"Audio pre-fetch failed for {file_name}: {e}")
            return f"\n\nNote: automatic audio transcription failed for {file_name}: {e}"

    def _enrich_question_text(self, item):
        task_id = item.task_id
        file_name = item.file_name
        question_text = (
            f"{item.question} "
            "Think hard to answer. Parse all statements in the question to make a plan. "
            "Your final answer should be a number or as few words as possible. "
            "Only use abbreviations when the question calls for abbreviations. "
            "If needed, use a comma separated list of values; the comma is always followed by a space. "
            f"Critically review your answer before making it the final answer. "
            f"Double check the answer to make sure it meets all format requirements stated in the question. "
            f"task_id: {task_id}."
        )

        # Deterministic pre-processing: don't rely on the agent remembering to
        # watch the video or listen to the audio -- do it now and hand it the
        # content directly as text.
        question_text += self._prefetch_video_description(item.question)

        if file_name:
            if file_name.lower().endswith(AUDIO_EXTENSIONS):
                question_text += self._prefetch_audio_transcript(task_id, file_name)
            else:
                question_text = f"{question_text} file_name: {file_name} (use tools to fetch the file)"

        return question_text

    async def _run_agent_async(self, item: Question):
        """Runs the agent asynchronously."""
        task_id = item.task_id
        question_text = await asyncio.to_thread(self._enrich_question_text, item)
        try:
            answer = await asyncio.to_thread(ManagerAgent(self.settings), question_text)
        except Exception as e:
            logger.error(f"Error running agent on task {task_id}: {e}")
            answer = f"AGENT ERROR: {e}"
        return QuestionAnswerPair(task_id=task_id,
                                  question=item.question, answer=str(answer))

    def _assign_questions(self, questions: list[Question]):
        """Runs the asynchronous loop and returns task outputs."""
        tasks = [self._run_agent_async(item) for item in questions]
        return asyncio.gather(*tasks)

    def run_agent(self, questions: list[Question], username: str) -> pd.DataFrame:
        """Run the agent(s) async, save answers and return a dataframe"""
        # Assign questions to agents and wait
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        def run_tasks_in_thread():
            question_answer_pairs = loop.run_until_complete(
                self._assign_questions(questions))
            return question_answer_pairs

        pairs = run_tasks_in_thread()

        # save json to disk and return a dataframe
        self._save_pairs(pairs, username)
        results_log = [pair.model_dump() for pair in pairs if pair is not None]
        if not results_log:
            logger.warning("Agent did not produce any answers to submit.")

        return pd.DataFrame(results_log)