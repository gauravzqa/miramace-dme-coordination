import importlib
import pkgutil

import miramace
from miramace.agents.schemas import CallerTurn
from miramace.models import CallTask
from miramace.ports import Brain, TranscriptView


class _EchoBrain:
    async def next(self, view: TranscriptView) -> CallerTurn:
        return CallerTurn(say="hi", slots_touched=[], action="continue", hangup_reason=None)


async def test_brain_protocol_and_view():
    task = CallTask(task_id="t", party="pcp", party_id="pcp", org_name="Sunrise Family Medicine",
                    phone="(312) 555-0198", intent="request_order", purpose="p", slots=[])
    view = TranscriptView(task=task, phase="opening", utterances=[], me="caller")
    brain = _EchoBrain()
    assert isinstance(brain, Brain)
    assert (await brain.next(view)).say == "hi"


def test_all_modules_import():
    for m in pkgutil.walk_packages(miramace.__path__, "miramace."):
        if m.name.startswith("miramace.voice."):
            continue
        importlib.import_module(m.name)
