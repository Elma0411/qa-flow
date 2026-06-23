from dataclasses import dataclass
from typing import Dict, Optional


COMMON_SYSTEM_PROMPT = """你是面向技术文档重建的多模态信息抽取助手。你的任务是把图片中的有效信息改写成可直接并入正文、适合后续问答知识抽取的中文文本块。

硬性要求：
1. 只输出最终正文，不输出分析过程、标题、说明语、客套话或格式提示。
2. 禁止输出 Markdown 表格、ASCII 表格、代码块、JSON、CSV、项目符号列表，禁止出现连续的 `|`、`---`、`+---+` 等表格结构字符。
3. 输出必须是自然中文句子。可以分成 1 至 3 段，但每段都必须是完整、连贯、可直接插入文档的正文。
4. 保留图中可确认的实体、参数、数值、单位、时间、状态、约束和关系；无法辨认时明确说明“图中该处文字无法辨认”，不要臆造。
5. 不重复上文下文已经明确表达的内容，只补充图片独有且对理解文档有价值的信息。
6. 直接陈述事实，不要写“该图展示了”“如下表所示”“见下图”等空泛引导语。
7. 如果图片中包含表格，也必须把表格内容改写成自然语言，而不是按行列抄写。
"""


REWRITE_SYSTEM_PROMPT = """你是技术文档清洗助手。你的任务是把已有的图片转写结果重写为适合知识抽取的中文正文。

硬性要求：
1. 删除所有表格结构、列分隔符、代码块和机械的行列复述。
2. 保留原文中可以确认的事实，不补造未出现的信息。
3. 输出自然、客观、信息密集，便于下游按文本块进行问答和知识提取。
4. 只输出重写后的正文。
"""


COMMON_REQUIREMENTS = """通用写作要求：
1. 优先抽取图片中独有的关键信息，例如主体、动作、关系、流程、参数、结果、异常、约束或结论。
2. 输出要与上下文自然衔接，但不要把上下文原文重复一遍。
3. 如果信息较多，按语义分组组织句子，不要按视觉顺序机械逐格、逐框、逐箭头念出来。
4. 只输出最终正文。"""


TABLE_REQUIREMENTS = """这是一张表格、清单、报表或表单图片。请在不保留表格结构的前提下，把它改写成适合知识抽取的中文正文。输出应包含两类信息：先给出表格摘要，再给出表格事实；两者都必须写成自然中文句子，不要使用标题、列表、Markdown 表格、CSV、JSON 或行列结构。

表格处理规则：
1. 先判断表格状态：空表或模板表、已填报表、部分填写表格、嵌套或分组表格、信息不可完整辨认的表格。
2. 表格摘要应概括表格的用途、适用对象、记录主题、统计口径、时间范围、单位、审批或管理场景，以及表格整体表达出的结论、趋势、异常、汇总结果或责任关系。不要重复上下文已经明确写出的内容。
3. 表格事实应尽量按“每个有效记录行一条事实”的方式转写，但必须写成自然句。每条事实要以该行的核心主体、项目名称、编号、类别、时间或其他可区分字段开头，合并该行中可确认的指标、数值、单位、状态、结果、备注和对应关系。不要写“第 1 行”“第 2 列”，不要机械拼接列名。
4. 如果表格存在多级表头、合并单元格、分组行、父子项、嵌套小表或横向分栏，应先识别上级分组、共同限定条件和子项关系。转写行事实时，把上级分组或共同条件并入对应事实，避免丢失层级含义。小计、合计、平均值、排名、阈值、判定结果和异常标记应作为独立事实或并入相关事实中说明。
5. 如果是空表、模板表或只有表头而没有具体填写内容，应说明它用于记录什么、需要填写哪些信息类别、有哪些填写顺序、约束条件、单位、频次、审核签字或备注要求。不要逐项罗列所有空白字段，不要输出空表骨架，也不要编造未填写的事实。
6. 如果是部分填写表格，应先概括表格用途和应包含的信息范围，再陈述已经填写的行事实，最后说明仍可识别但未填写的信息类别或约束。空白单元格只在影响理解时说明为未填写，不要逐格描述空白。
7. 如果文字、数值或单位局部无法辨认，应保留可确认部分，并明确说明“图中该处文字无法辨认”或“该数值无法辨认”。不要根据上下文猜测缺失内容。
8. 如果表格行数很多，应优先完整保留非空记录、异常记录、合计汇总、关键指标和有判定结论的记录；对重复性很强且无法逐行完整输出的部分，可以按类别汇总，但要说明可见记录呈现的共同事实和缺失范围。
9. 如果表格中含有签字、盖章、勾选、删除线、手写备注、附件编号、脚注或特殊符号，应在其影响审批状态、责任归属、结论判断或数据解释时转写为自然事实。
10. 输出 1 至 3 段自然中文，直接给出正文。第一部分表达表格摘要，后续句子表达按行细分的表格事实；不得输出 `|`、制表符表格、Markdown 表格、CSV、行列标题拼接串、项目符号列表或代码块。"""


CONTEXT_TEMPLATE = """上文：{a_context}
下文：{b_context}

{task_instructions}

{common_requirements}
"""


TABLE_REWRITE_PROMPT_TEMPLATE = """上文：{a_context}
下文：{b_context}

原始转写结果：
{raw_text}

请将以上内容重写为可直接并入技术文档正文的中文文本块，并满足以下要求：
1. 删除所有表格结构、分隔符、列标题拼接和机械的行列复述。
2. 输出应包含表格摘要和表格事实两类信息，但都必须写成自然中文正文，不要使用显式标题、列表或表格。
3. 表格摘要应概括表格用途、记录对象、统计口径、时间范围、单位、管理场景，以及可确认的整体结论、趋势、异常、汇总结果或责任关系。
4. 如果内容对应已填报表，尽量按每个有效记录行转写为一条自然事实，保留该行可确认的主体、项目、编号、类别、时间、指标、数值、单位、状态、结果、备注和对应关系。
5. 如果内容对应嵌套表格、多级表头、合并单元格、分组行或父子项，把上级分组、共同限定条件和子项关系并入对应事实，避免丢失层级含义。
6. 如果内容明显对应空表、模板表或只有表头没有具体填写内容，说明这张表用于记录什么、需要填写哪些信息类别、有哪些填写顺序、约束、单位、频次或审批信息，不要罗列空白字段。
7. 如果内容对应部分填写的表格，先概括用途，再陈述已填行事实，最后补充仍可识别的未填字段类别或约束。
8. 不要重复上下文已经明确写出的内容，不要补造事实；无法辨认的文字或数值应明确说明无法辨认。
9. 只输出重写后的正文。"""


def _build_prompt(task_instructions: str) -> str:
    return CONTEXT_TEMPLATE.format(
        a_context="{a_context}",
        b_context="{b_context}",
        task_instructions=task_instructions,
        common_requirements=COMMON_REQUIREMENTS,
    )


UNIFIED_PROMPT_TEMPLATE = _build_prompt(
    """请根据图片内容生成可直接插入正文的中文文本。
1. 如果图片属于流程图、结构图、示意图或电路图，重点说明组成部分、连接关系、信号或数据流向、关键参数、约束条件以及图中独有的结论。
2. 如果图片包含表格，请按“空表/模板表、已填报表、部分填写表格”三种情况之一转写为自然中文，不要保留行列结构。
3. 如果图片主要是签章、签名或审批痕迹，说明可辨认的机构、人员、日期、动作和它们与正文的对应关系。"""
)


SHEET_PROMPT_TEMPLATE = _build_prompt(TABLE_REQUIREMENTS)


CIRCUIT_PROMPT_TEMPLATE = _build_prompt(
    """这是一张电路图或接线图。请提取对文档理解最关键的信息：
1. 说明关键器件、接口、节点、型号、参数和标注。
2. 说明连接关系、信号流向、输入输出、供电关系以及必要的工作条件。
3. 如果图中存在保护、控制、时序、反馈或联锁关系，要明确写出来。
4. 不要只罗列器件名称，要把器件和关系组织成可读的正文。"""
)


SEAL_PROMPT_TEMPLATE = _build_prompt(
    """这是一张签章、签名或审批痕迹图片。请提取可确认的信息：
1. 识别印章文字、印章类型、签名人、日期、位置和与正文的对应关系。
2. 如果能看出审核、批准、盖章、签收、会签等动作或流程状态，也要明确写出。
3. 无法辨认的局部可说明无法辨认，但不要凭空猜测身份、机构或日期。"""
)


ARCHITECTURE_PROMPT_TEMPLATE = _build_prompt(
    """这是一张架构图、模块图或系统关系图。请提取对正文最重要的信息：
1. 说明主要模块、层级、职责、外部系统和边界。
2. 说明模块之间的数据流、调用链、依赖关系、部署关系或控制关系。
3. 如果图中体现输入输出、协议、接口、同步异步或容错机制，要明确写出。
4. 输出应是可直接合并进正文的说明文本，而不是框图标签清单。"""
)


SCHEMATIC_PROMPT_TEMPLATE = _build_prompt(
    """这是一张示意图、流程图或结构示意图。请提取核心信息：
1. 说明对象、部件、步骤、空间关系、方向、标注和关键文字。
2. 如果图中表达某种原理、工艺过程、操作流程或结构关系，要按逻辑顺序转写成句子。
3. 保留图中可确认的参数、编号、图例和条件，不要机械复述箭头或框名。"""
)


MACRO_LINE_OVERVIEW_PROMPT_TEMPLATE = _build_prompt(
    """这是一张宏观线路总图、系统总图或全局连接关系图。请提取对正文最重要的总体信息：
1. 说明图中的系统范围、主干线路、分支线路、上下游对象、关键节点、接口和边界。
2. 说明线路或信息、能量、物料、控制信号的总体流向，以及各区域、设备、模块之间的连接关系。
3. 保留图例、编号、区域名称、回路名称、容量、规格、方向、状态、约束条件和跨页引用等可确认信息。
4. 不要陷入局部元件清单；优先概括整体拓扑、主从关系、分区关系和对理解文档有价值的异常或结论。"""
)


LOCAL_CIRCUIT_PROMPT_TEMPLATE = _build_prompt(
    """这是一张局部电路原理图、接线细节图或局部回路图。请提取可直接并入正文的技术事实：
1. 说明关键器件、端子、引脚、节点、网络名、型号、参数、供电和接地关系。
2. 说明输入输出、信号流向、控制关系、保护关系、反馈关系、联锁关系、时序关系和触发条件。
3. 对可辨认的阻容值、电压电流、接口编号、线号、开关状态、测试点和注释要准确保留。
4. 不要只罗列元件名称，要把元件、连接和工作逻辑组织成连贯正文；无法辨认的标注要明确说明。"""
)


DATA_TABLE_PROMPT_TEMPLATE = _build_prompt(
    """这是一张已经填写数据的表格、清单、报表或统计表。请在不保留表格结构的前提下，把表中事实改写成自然中文正文。
1. 先概括表格的记录对象、统计口径、时间范围、单位、版本、适用范围和整体结论或异常。
2. 对每条有效记录，保留主体、项目、编号、类别、时间、指标、数值、单位、状态、判定结果、备注和对应关系。
3. 多级表头、合并单元格、分组项、小计、合计、平均值、排名、阈值和异常标记必须并入对应事实，避免丢失层级含义。
4. 行数较多时，优先保留非空记录、异常记录、合计汇总、关键指标和判定结论；重复内容可归纳但不能编造。"""
)


TEXT_RECORD_TABLE_PROMPT_TEMPLATE = _build_prompt(
    """这是一张以文字记录为主的表单、台账、检查记录、会议记录、审批记录或描述性记录表。请提取自然中文事实：
1. 说明记录主题、适用对象、记录时间、责任单位、责任人、检查或处理场景、审批流转状态和结论。
2. 将字段和值、勾选项、手写补充、签字盖章、备注和附件编号改写成完整句子，不要按表格栏位机械复述。
3. 如果包含问题描述、整改措施、处理意见、验收结论或风险提示，要保留责任主体、动作、期限、结果和状态。
4. 未填写字段只在影响理解时说明；无法辨认的文字要明确说明，不能猜测姓名、日期或结论。"""
)


GANTT_CHART_PROMPT_TEMPLATE = _build_prompt(
    """这是一张甘特图、进度计划图或任务排期图。请提取计划与执行关系：
1. 说明项目或计划范围、时间轴单位、起止时间、阶段划分、里程碑和当前状态。
2. 按任务或阶段转写可确认的开始时间、结束时间、持续时间、负责人或责任单位、前后依赖、并行关系和关键路径。
3. 保留延期、提前、完成率、基线对比、风险节点、验收节点和异常标记等对进度判断有价值的信息。
4. 不要输出表格或逐格描述时间条；应把进度关系写成连贯正文。"""
)


FLOW_ARCHITECTURE_PROMPT_TEMPLATE = _build_prompt(
    """这是一张流程图、架构图、模块关系图或结构示意图。请提取图中表达的逻辑关系：
1. 说明主要模块、步骤、对象、层级、职责、外部系统、边界和输入输出。
2. 说明数据流、业务流、控制流、调用链、依赖关系、部署关系、条件分支、反馈闭环和异常处理路径。
3. 保留接口、协议、编号、状态、条件、参数、图例和关键注释。
4. 输出应是可直接合并进正文的说明文本，不要变成框名、箭头或节点标签清单。"""
)


BLANK_GENERIC_TABLE_PROMPT_TEMPLATE = _build_prompt(
    """这是一张空白通用表格、模板表、待填写表单或只有表头的记录表。请说明表格用途和填写约束，不要编造未填写事实：
1. 判断表格用于记录什么对象、业务环节、统计口径、检查项目、审批流程或管理场景。
2. 概括需要填写的信息类别，例如主体信息、编号、时间、指标、状态、结果、责任人、签字、备注、附件或审核意见。
3. 保留可见的单位、频次、填写顺序、必填关系、分组层级、审批签章位置和备注要求。
4. 不要逐项罗列所有空白字段，不要输出表格骨架；如果只有表头没有数据，应明确说明未见具体填写内容。"""
)


EQUIPMENT_LAYOUT_PROMPT_TEMPLATE = _build_prompt(
    """这是一张设备布局图、平面布置图、安装位置图或现场空间关系图。请提取空间与设备关系：
1. 说明设备、柜体、管线、通道、房间、区域、方向、编号和图例所表达的对象。
2. 说明设备之间的位置关系、连接关系、安装关系、相邻关系、进出线方向、物流或人员动线，以及与墙体、门窗、边界的关系。
3. 保留可辨认的尺寸、坐标、标高、间距、安全距离、安装要求、容量、规格、方位和约束条件。
4. 不要只罗列设备名称，要把布局关系和对施工、运维或理解系统有价值的信息写成连贯正文。"""
)


@dataclass(frozen=True)
class PromptConfig:
    category_key: str
    display_name: str
    system_prompt: str
    prompt_template: str


DEFAULT_PROMPT_KEY = "others"

# Previous 6-class-only routing kept for reference:
#
# PROMPT_CONFIGS: Dict[str, PromptConfig] = {
#     "sheet": PromptConfig("sheet", "表格", COMMON_SYSTEM_PROMPT, SHEET_PROMPT_TEMPLATE),
#     "circuit": PromptConfig("circuit", "电路图", COMMON_SYSTEM_PROMPT, CIRCUIT_PROMPT_TEMPLATE),
#     "seal": PromptConfig("seal", "签章签名", COMMON_SYSTEM_PROMPT, SEAL_PROMPT_TEMPLATE),
#     "architecture": PromptConfig("architecture", "架构图", COMMON_SYSTEM_PROMPT, ARCHITECTURE_PROMPT_TEMPLATE),
#     "schematic": PromptConfig("schematic", "示意图", COMMON_SYSTEM_PROMPT, SCHEMATIC_PROMPT_TEMPLATE),
#     DEFAULT_PROMPT_KEY: PromptConfig(DEFAULT_PROMPT_KEY, "其他", COMMON_SYSTEM_PROMPT, UNIFIED_PROMPT_TEMPLATE),
# }

PROMPT_CONFIGS: Dict[str, PromptConfig] = {
    "macro_line_overview": PromptConfig(
        category_key="macro_line_overview",
        display_name="宏观线路总图",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=MACRO_LINE_OVERVIEW_PROMPT_TEMPLATE,
    ),
    "local_circuit_schematic": PromptConfig(
        category_key="local_circuit_schematic",
        display_name="局部电路原理图",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=LOCAL_CIRCUIT_PROMPT_TEMPLATE,
    ),
    "data_table": PromptConfig(
        category_key="data_table",
        display_name="数据表格",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=DATA_TABLE_PROMPT_TEMPLATE,
    ),
    "text_record_table": PromptConfig(
        category_key="text_record_table",
        display_name="文本描述性记录表",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=TEXT_RECORD_TABLE_PROMPT_TEMPLATE,
    ),
    "gantt_chart": PromptConfig(
        category_key="gantt_chart",
        display_name="甘特图",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=GANTT_CHART_PROMPT_TEMPLATE,
    ),
    "flow_architecture_diagram": PromptConfig(
        category_key="flow_architecture_diagram",
        display_name="示意图（流程图／架构图）",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=FLOW_ARCHITECTURE_PROMPT_TEMPLATE,
    ),
    "blank_generic_table": PromptConfig(
        category_key="blank_generic_table",
        display_name="空白通用表格",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=BLANK_GENERIC_TABLE_PROMPT_TEMPLATE,
    ),
    "equipment_layout": PromptConfig(
        category_key="equipment_layout",
        display_name="设备布局图",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=EQUIPMENT_LAYOUT_PROMPT_TEMPLATE,
    ),
    "sheet": PromptConfig(
        category_key="sheet",
        display_name="表格",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=SHEET_PROMPT_TEMPLATE,
    ),
    "circuit": PromptConfig(
        category_key="circuit",
        display_name="电路图",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=CIRCUIT_PROMPT_TEMPLATE,
    ),
    "seal": PromptConfig(
        category_key="seal",
        display_name="签章签名",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=SEAL_PROMPT_TEMPLATE,
    ),
    "architecture": PromptConfig(
        category_key="architecture",
        display_name="架构图",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=ARCHITECTURE_PROMPT_TEMPLATE,
    ),
    "schematic": PromptConfig(
        category_key="schematic",
        display_name="示意图",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=SCHEMATIC_PROMPT_TEMPLATE,
    ),
    DEFAULT_PROMPT_KEY: PromptConfig(
        category_key=DEFAULT_PROMPT_KEY,
        display_name="其他",
        system_prompt=COMMON_SYSTEM_PROMPT,
        prompt_template=UNIFIED_PROMPT_TEMPLATE,
    ),
}


def get_prompt_config(category_key: Optional[str], enable_classification: bool = True) -> PromptConfig:
    if not enable_classification:
        return PROMPT_CONFIGS[DEFAULT_PROMPT_KEY]
    return PROMPT_CONFIGS.get(category_key or DEFAULT_PROMPT_KEY, PROMPT_CONFIGS[DEFAULT_PROMPT_KEY])
