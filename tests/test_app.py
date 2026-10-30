from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_sample_page_runs_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    assert len(app.tabs) == 5
    assert [title.value for title in app.title] == ["投资组合优化与风险研究平台"]
    assert [button.label for button in app.button] == ["运行组合优化", "运行滚动回测"]
    assert [radio.label for radio in app.radio] == ["优化模式"]
    assert app.radio[0].options == ["最小方差", "目标收益率最小方差", "有效前沿"]


def test_target_return_mode_runs_solver_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=120).run()

    app.radio[0].set_value("目标收益率最小方差").run()
    assert "目标年化预期收益率（%）" in [item.label for item in app.number_input]

    app.button[0].click().run(timeout=120)

    assert not app.exception
    assert any("求解器：" in item.value for item in app.caption)


def test_efficient_frontier_mode_renders_frontier_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=120).run()

    app.radio[0].set_value("有效前沿").run()
    app.button[0].click().run(timeout=120)

    assert not app.exception
    assert "有效前沿" in [item.value for item in app.subheader]


def test_rolling_backtest_page_runs_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=120).run()

    backtest_button = next(button for button in app.button if button.label == "运行滚动回测")
    backtest_button.click().run(timeout=120)

    assert not app.exception
    assert "样本外回测结果" in [item.value for item in app.subheader]
