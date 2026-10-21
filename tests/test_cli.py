import pytest
import typer
from typer.testing import CliRunner

from edurec.cli.utils import parse_seeds
from edurec.main import app


def test_cli_help_lists_main_commands():
    result = CliRunner().invoke(app, ["--help"], color=False)
    assert result.exit_code == 0
    for command in ("train", "eval", "dataset", "hpo", "ablation"):
        assert command in result.output


def test_parse_seeds_accepts_spaces_and_rejects_empty_input():
    assert parse_seeds("42, 7, , 13") == [42, 7, 13]
    with pytest.raises(typer.BadParameter, match="At least one seed"):
        parse_seeds(" , ")
