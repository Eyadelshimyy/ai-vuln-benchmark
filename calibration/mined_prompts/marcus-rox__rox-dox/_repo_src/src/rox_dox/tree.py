from __future__ import annotations

from dataclasses import dataclass

from rox_dox.model import Page


@dataclass(frozen=True)
class SiteTree:
    pages: dict[str, Page]
    children: dict[str, list[str]]
    root: str

    def ancestors(self, page_id: str) -> list[str]:
        ancestors = []
        parent_id = self.pages[page_id].parent
        while parent_id is not None:
            ancestors.append(parent_id)
            parent_id = self.pages[parent_id].parent
        return list(reversed(ancestors))

    def children_of(self, page_id: str) -> list[str]:
        return self.children[page_id]


def tree_problems(pages: list[Page]) -> list[str]:
    problems = []
    pages_by_id: dict[str, Page] = {}
    for page in pages:
        if page.id in pages_by_id:
            problems.append(f"duplicate page id '{page.id}'")
        else:
            pages_by_id[page.id] = page

    page_ids = set(pages_by_id)
    roots = [page for page in pages if page.parent is None]
    if len(roots) != 1:
        problems.append(f"expected exactly one root page, found {len(roots)}")

    for page in pages:
        if page.parent is not None and page.parent not in page_ids:
            problems.append(f"page '{page.id}' has missing parent '{page.parent}'")

    visited = set()
    for start_id in sorted(pages_by_id):
        chain = []
        positions = {}
        current_id = start_id
        cycle_start = None
        while current_id in pages_by_id and current_id not in visited:
            if current_id in positions:
                cycle_start = positions[current_id]
                break
            positions[current_id] = len(chain)
            chain.append(current_id)
            parent_id = pages_by_id[current_id].parent
            if parent_id is None or parent_id not in pages_by_id:
                break
            current_id = parent_id
        if cycle_start is not None:
            cycle = chain[cycle_start:]
            problems.append(f"parent cycle: {' -> '.join([*cycle, cycle[0]])}")
        visited.update(chain)

    for page in pages:
        for node in page.block.nodes:
            if node.link is not None and node.link not in page_ids:
                problems.append(
                    f"block node {node.id}: link to missing page '{node.link}'"
                )
        for related in page.related:
            if related.page is not None and related.page not in page_ids:
                problems.append(
                    f"related '{related.label}': link to missing page '{related.page}'"
                )
        for domain in page.data.domains:
            if domain.page is not None and domain.page not in page_ids:
                problems.append(
                    f"schema domain '{domain.id}': link to missing page '{domain.page}'"
                )
            elif domain.page is not None:
                expected_kind = {"root": "domain", "domain": "feature"}.get(page.kind)
                target_kind = pages_by_id[domain.page].kind
                if expected_kind is not None and target_kind != expected_kind:
                    problems.append(
                        f"schema domain '{domain.id}': page '{domain.page}' has "
                        f"kind '{target_kind or 'unspecified'}'; expected "
                        f"'{expected_kind}'"
                    )
        for table in [
            page.tldr.table,
            *(page.tldr.additional_tables or []),
        ]:
            for link in table.links:
                if link.page not in page_ids:
                    problems.append(
                        f"TLDR table: link to missing page '{link.page}'"
                    )

    for page in pages:
        if page.parent is None:
            if page.paths != ["."]:
                problems.append(f"root page '{page.id}' paths must be exactly ['.']")
            if page.kind != "root":
                problems.append(
                    f"page '{page.id}' has kind '{page.kind}' with parent kind "
                    "'none'; root pages must have kind 'root'"
                )
            continue

        parent = pages_by_id.get(page.parent)
        parent_kind = parent.kind if parent is not None else "missing"
        if page.kind == "root":
            problems.append(
                f"page '{page.id}' has kind 'root' but parent '{page.parent}' "
                f"has kind '{parent_kind}'"
            )
        elif page.kind == "domain" and parent_kind != "root":
            problems.append(
                f"page '{page.id}' has kind 'domain' but parent '{page.parent}' "
                f"has kind '{parent_kind}'; domains require a root parent"
            )
        elif page.kind == "feature" and parent_kind not in {"domain", "feature"}:
            problems.append(
                f"page '{page.id}' has kind 'feature' but parent '{page.parent}' "
                f"has kind '{parent_kind}'; features require a domain or feature parent"
            )

    return problems


def build_tree(pages: list[Page]) -> SiteTree:
    pages_by_id = {page.id: page for page in pages}
    children = {page_id: [] for page_id in pages_by_id}
    root = next(page.id for page in pages if page.parent is None)
    for page in pages:
        if page.parent is not None:
            children[page.parent].append(page.id)
    for child_ids in children.values():
        child_ids.sort(key=lambda page_id: (pages_by_id[page_id].title, page_id))
    return SiteTree(pages=pages_by_id, children=children, root=root)
