from .backend import query
from .journal import Journal
from .utils.config import StageConfig
from ai_scientist.utils.prompt_manager import get_prompt


def journal2report(journal: Journal, task_desc: dict, rcfg: StageConfig):
    """
    Generate a report from a journal, the report will be in markdown format.
    """
    report_input = journal.generate_summary(include_code=True)
    # 从 prompt.yaml 读取 Journal2Report 系统提示词
    system_prompt_dict = get_prompt('journal2report.system_prompt')
    # 从 prompt.yaml 读取 Journal2Report 上下文提示词并格式化
    context_prompt = get_prompt(
        'journal2report.context_prompt',
        report_input=report_input,
        task_desc=task_desc
    )
    return query(
        system_message=system_prompt_dict,
        user_message=context_prompt,
        model=rcfg.model,
        temperature=rcfg.temp,
    )
