"""Shared rich console widgets."""
from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

console = Console()


def edit_kv(title: str, fields: dict[str, str], *, help: dict[str, str] | None = None) -> dict[str, str]:
    """Interactive key-value editor.

    Renders `fields` as a numbered table; the user types a row number to edit
    that value (current value is the prompt default), or Enter to accept all.
    Returns the (possibly edited) dict. Pure rich -- works over SSH, no TTY
    framework needed.
    """
    fields = dict(fields)
    help = help or {}
    keys = list(fields)
    while True:
        table = Table(title=title, title_justify="left")
        table.add_column("#", style="cyan", justify="right")
        table.add_column("field")
        table.add_column("value")
        for i, k in enumerate(keys, 1):
            val = fields[k] if fields[k] != "" else "[dim](empty)[/dim]"
            table.add_row(str(i), k, val)
        console.print(table)
        choice = Prompt.ask("  edit # (Enter to accept all)", default="", show_default=False)
        choice = choice.strip()
        if not choice:
            return fields
        if not choice.isdigit() or not (1 <= int(choice) <= len(keys)):
            console.print("  [red]enter a row number, or Enter to finish[/red]")
            continue
        k = keys[int(choice) - 1]
        hint = help.get(k, "")
        label = f"  {k}" + (f"  [dim]{hint}[/dim]" if hint else "")
        fields[k] = Prompt.ask(label, default=fields[k])
