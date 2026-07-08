from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .io import read_text, stable_json_hash, write_text


PLUGIN_REL_PATH = ".llmwiki/page_plugin.yaml"


class RelatedPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    same_ingest_model_max: int = Field(default=5, ge=0)
    embedding_existing_top_k: int = Field(default=3, ge=0)
    embedding_existing_min_similarity: float = Field(default=0.7, ge=0.0, le=1.0)
    replacement_margin: float = Field(default=0.04, ge=0.0, le=1.0)


class PageTypeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str
    title_prefix: str
    label: str
    use_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    section_guidance: dict[str, str] = Field(default_factory=dict)
    writing_rules: list[str] = Field(default_factory=list)

    @field_validator("directory", "title_prefix", "label")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("不能为空。")
        return value.strip()


class PagePlugin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str
    default_page_type: str
    page_types: dict[str, PageTypeSpec]
    related: RelatedPolicy = Field(default_factory=RelatedPolicy)
    body_wikilink_limit: int = Field(default=2, ge=0)

    @field_validator("name", "version", "description", "default_page_type")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("不能为空。")
        return value.strip()

    def page_type(self, page_type: str) -> PageTypeSpec:
        if page_type not in self.page_types:
            raise KeyError(f"未知页面类型：{page_type}")
        return self.page_types[page_type]

    def page_type_or_default(self, page_type: str) -> tuple[str, PageTypeSpec]:
        if page_type in self.page_types:
            return page_type, self.page_types[page_type]
        return self.default_page_type, self.page_types[self.default_page_type]

    def page_type_for_path(self, rel_path: str) -> str:
        for page_type, spec in self.page_types.items():
            if rel_path.startswith(f"{spec.directory}/"):
                return page_type
        return self.default_page_type

    def knowledge_directories(self) -> set[str]:
        return {spec.directory for spec in self.page_types.values()}


DEFAULT_SECTIONS = ["摘要", "核心内容", "矛盾与未解决问题"]
DEFAULT_SECTION_GUIDANCE = {
    "摘要": "用 2-4 句话说明这页长期维护的对象、价值和边界。",
    "核心内容": "按读者理解顺序组织定义、机制、结构、方法、证据和限制；必要时使用更具体的三级小节。",
    "矛盾与未解决问题": "只记录来源中真实存在的张力、限制、争议、未回答问题或后续待确认点；没有则写“暂无明确未解决问题。”。",
}


DEFAULT_PAGE_PLUGIN = {
    "name": "default",
    "version": "1",
    "description": "llmwiki Lite 默认页面插件，面向 100 页以内的轻量中文知识库。",
    "default_page_type": "concept",
    "body_wikilink_limit": 2,
    "related": {
        "same_ingest_model_max": 5,
        "embedding_existing_top_k": 3,
        "embedding_existing_min_similarity": 0.7,
        "replacement_margin": 0.04,
    },
    "page_types": {
        "concept": {
            "directory": "concepts",
            "title_prefix": "Concept_",
            "label": "概念",
            "use_when": [
                "抽象概念、机制、原则、能力模型、方法论或可长期复用的知识对象。",
                "raw 的核心价值是解释“是什么、为什么重要、如何运作”。",
            ],
            "avoid_when": [
                "内容主体是某个产品、组织、人物或项目实体。",
                "内容主体是完整方案、系统设计、对比或总览。",
            ],
            "sections": DEFAULT_SECTIONS,
            "section_guidance": DEFAULT_SECTION_GUIDANCE,
            "writing_rules": [
                "不要把单个事实、工具参数、安装步骤或边缘例子抬成概念页。",
                "概念页应解释稳定内涵和适用边界，而不是堆砌来源摘录。",
            ],
        },
        "overview": {
            "directory": "overviews",
            "title_prefix": "Overview_",
            "label": "总览",
            "use_when": [
                "raw 的主体是产品、平台、SDK、框架、领域、项目或能力体系的整体介绍。",
                "页面需要帮助读者先建立全局地图，再理解组成部分、适用场景和边界。",
            ],
            "avoid_when": [
                "内容只解释一个单一概念或机制。",
                "内容主体是具体方案实现、实体档案或对比判断。",
            ],
            "sections": DEFAULT_SECTIONS,
            "section_guidance": DEFAULT_SECTION_GUIDANCE,
            "writing_rules": [
                "总览页要优先说明对象边界、组成模块、使用场景和阅读入口。",
                "不要把总览页写成松散资料汇编；必须给出可维护的结构。",
            ],
        },
        "design": {
            "directory": "designs",
            "title_prefix": "Design_",
            "label": "设计",
            "use_when": [
                "raw 的主体是方案、架构、流程、协议、实现策略或设计取舍。",
                "页面需要描述问题、设计目标、关键机制、权衡和落地约束。",
            ],
            "avoid_when": [
                "内容只是抽象概念介绍。",
                "内容只是产品总览、实体档案或多对象比较。",
            ],
            "sections": DEFAULT_SECTIONS,
            "section_guidance": DEFAULT_SECTION_GUIDANCE,
            "writing_rules": [
                "设计页必须保留关键约束和取舍，不要只写成最佳实践列表。",
                "如果设计只是一段附属材料，应并入主干页而非独立成页。",
            ],
        },
        "entity": {
            "directory": "entities",
            "title_prefix": "Entity_",
            "label": "实体",
            "use_when": [
                "raw 的主体是人物、组织、产品、项目、论文、模型、公司或明确命名对象。",
                "页面需要维护这个实体的定位、能力、事实和与其他页面的关系。",
            ],
            "avoid_when": [
                "实体只是案例或证据，不是 raw 的长期知识对象。",
                "内容更适合写成概念、总览、设计或对比。",
            ],
            "sections": DEFAULT_SECTIONS,
            "section_guidance": DEFAULT_SECTION_GUIDANCE,
            "writing_rules": [
                "实体页要区分事实、定位和边界，不要把人物背景或公司新闻过度展开。",
                "如果实体只是支撑主干观点，应作为段落进入主干页。",
            ],
        },
        "comparison": {
            "directory": "comparisons",
            "title_prefix": "Comparison_",
            "label": "对比",
            "use_when": [
                "raw 的主体是两个或多个对象、方法、产品、模型或方案之间的差异、取舍和选择。",
                "页面需要帮助读者理解何时选择 A、何时选择 B，以及背后的判断标准。",
            ],
            "avoid_when": [
                "raw 只是顺带提到多个对象，没有形成明确比较。",
                "内容实际是在讲单一概念、总览、设计或实体。",
            ],
            "sections": DEFAULT_SECTIONS,
            "section_guidance": DEFAULT_SECTION_GUIDANCE,
            "writing_rules": [
                "对比页必须围绕判断维度组织，而不是分别罗列对象介绍。",
                "若比较只是某主干页里的一个局部论点，应降级为段落。",
            ],
        },
    },
}


def default_page_plugin() -> PagePlugin:
    return PagePlugin.model_validate(DEFAULT_PAGE_PLUGIN)


def page_plugin_path(vault: Path) -> Path:
    return vault.expanduser().resolve() / PLUGIN_REL_PATH


def load_page_plugin(vault: Path) -> PagePlugin:
    path = page_plugin_path(vault)
    if not path.exists():
        raise ValueError("Lite vault 缺少 .llmwiki/page_plugin.yaml，请先运行 llmwiki init，或手动创建页面插件。")
    return PagePlugin.model_validate(yaml.safe_load(read_text(path)))


def write_page_plugin(vault: Path, plugin: PagePlugin | None = None) -> None:
    path = page_plugin_path(vault)
    payload = (plugin or default_page_plugin()).model_dump(mode="json")
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    write_text(path, text)


def page_plugin_hash(plugin: PagePlugin) -> str:
    return stable_json_hash(plugin.model_dump(mode="json"))


def page_plugin_taxonomy_card(plugin: PagePlugin) -> str:
    lines = [
        f"页面插件：{plugin.name}@{plugin.version}",
        f"默认页面类型：{plugin.default_page_type}",
        "允许页面类型：",
    ]
    for page_type, spec in plugin.page_types.items():
        lines.append(f"- {page_type}（{spec.label}）：目录 `{spec.directory}/`，文件名前缀 `{spec.title_prefix}`。")
        if spec.use_when:
            lines.append("  - 适用：" + "；".join(spec.use_when))
        if spec.avoid_when:
            lines.append("  - 避免：" + "；".join(spec.avoid_when))
    return "\n".join(lines)


def page_plugin_writing_card(plugin: PagePlugin, page_type: str) -> str:
    normalized_type, spec = plugin.page_type_or_default(page_type)
    lines = [
        f"页面类型：{normalized_type}（{spec.label}）",
        f"目标目录：`{spec.directory}/`",
        f"文件名前缀：`{spec.title_prefix}`",
        "固定章节顺序：" + " -> ".join(spec.sections),
        f"正文 wikilink 上限：{plugin.body_wikilink_limit} 条",
    ]
    if spec.writing_rules:
        lines.append("写作规则：")
        lines.extend(f"- {rule}" for rule in spec.writing_rules)
    if spec.section_guidance:
        lines.append("章节要求：")
        for section in spec.sections:
            guidance = spec.section_guidance.get(section, "")
            if guidance:
                lines.append(f"- {section}：{guidance}")
    return "\n".join(lines)


def page_plugin_related_card(plugin: PagePlugin) -> str:
    related = plugin.related
    return "\n".join(
        [
            "相关页面策略：",
            f"- 同一轮 ingest 中由模型提出的相关页最多保留 {related.same_ingest_model_max} 篇。",
            f"- 旧 wiki 页面由 embedding 计算，最多保留 top {related.embedding_existing_top_k} 篇。",
            f"- embedding 相关页相似度必须严格大于 {related.embedding_existing_min_similarity:.2f}。",
            "- Related 目标只能是知识页，不能是 raw、sources、logs、index 或 Source_*。",
        ]
    )
