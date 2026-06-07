from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any, Dict, Iterable

import pandas as pd
import streamlit as st

from financial_model import (
    DEFAULT_HALLS,
    DEFAULT_OPEX,
    ModelInputs,
    calculate_model,
    create_excel_report,
    summarize_metrics,
    _years_label,
)

st.set_page_config(page_title="Finanční analýza pro vodohospodářskou investici", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 6.2rem !important; padding-bottom: 2rem; max-width: 100%;}
    div[data-testid="stMetric"] {
        background: #F7F9FC;
        border: 1px solid #E4E8F0;
        padding: 10px 12px;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .app-note {
        background: #EEF5FF;
        border-left: 5px solid #1F4E79;
        padding: 10px 14px;
        border-radius: 8px;
        margin: 6px 0 14px 0;
    }
    .compact-note {
        color: #475569;
        font-size: 0.90rem;
        margin-top: -0.35rem;
        margin-bottom: 0.45rem;
    }
    div[data-testid="stDataFrame"] div[role="columnheader"] {font-size: 12px;}
    div[data-testid="stDataFrame"] div[role="gridcell"] {font-size: 12px;}
    .section-card {
        border: 1px solid #D8DEE9;
        border-radius: 12px;
        padding: 10px 12px 4px 12px;
        background: #FFFFFF;
        margin-bottom: 10px;
    }
    .input-topic-title {
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    div[data-testid="stTextInput"] input {
        min-height: 34px;
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
    }
    div[data-testid="stTextInput"] {margin-bottom: 0.05rem;}

    .pretty-table-wrap {
        border: 1px solid #D8DEE9;
        border-radius: 12px;
        overflow: hidden;
        margin: 0.35rem 0 0.8rem 0;
        background: #FFFFFF;
    }
    .pretty-table-wrap table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
    }
    .pretty-table-wrap th {
        background: #F1F5F9;
        color: #0F172A;
        font-weight: 700;
        padding: 8px 10px;
        border-bottom: 1px solid #D8DEE9;
        text-align: left;
        white-space: nowrap;
    }
    .pretty-table-wrap td {
        padding: 7px 10px;
        border-bottom: 1px solid #EEF2F7;
        vertical-align: middle;
    }
    .pretty-table-wrap tr:last-child td {border-bottom: none;}
    .pretty-table-wrap td.num {text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums;}
    .compact-pretty-table-wrap table {font-size: 0.86rem;}
    .compact-pretty-table-wrap th {padding: 5px 7px; white-space: normal; line-height: 1.15;}
    .compact-pretty-table-wrap td {padding: 4px 7px; line-height: 1.15;}
    .input-card {
        border: 1px solid #D8DEE9;
        border-radius: 12px;
        padding: 8px 10px 4px 10px;
        background: #FFFFFF;
        margin: 0.35rem 0 0.8rem 0;
    }
    .table-header {
        color: #64748B;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 0.15rem;
        white-space: normal;
        line-height: 1.2;
    }
    .unit-text {
        padding-top: 0.38rem;
        color: #475569;
        white-space: nowrap;
    }
    .app-title {
        display: block;
        position: relative;
        z-index: 10;
        width: 100%;
        box-sizing: border-box;
        font-size: clamp(1.35rem, 2.4vw, 2.2rem);
        line-height: 1.25;
        font-weight: 800;
        color: #0F172A;
        background: #FFFFFF;
        border: 1px solid #D8DEE9;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin: 1.1rem 0 1rem 0;
        white-space: normal;
        word-break: normal;
        overflow-wrap: anywhere;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .row-table {
        border: 1px solid #D8DEE9;
        border-radius: 12px;
        padding: 10px 12px;
        background: #FFFFFF;
        margin: 0.35rem 0 0.8rem 0;
    }
    .row-table .stButton > button {
        min-height: 34px;
        padding-left: 0.45rem;
        padding-right: 0.45rem;
    }
    .calculated-opex-row {
        background: #F8FAFC;
        border-top: 1px solid #D8DEE9;
        border-radius: 8px;
        padding: 6px 8px;
        margin-top: 4px;
        font-size: 0.92rem;
    }
    .calculated-opex-value {
        text-align: right;
        font-variant-numeric: tabular-nums;
        font-weight: 700;
        color: #0F172A;
        padding-top: 0.45rem;
        white-space: nowrap;
    }
    .calculated-opex-note {
        color: #475569;
        padding-top: 0.45rem;
    }

    .compact-wide-table-wrap {
        border: 1px solid #D8DEE9;
        border-radius: 12px;
        overflow-x: hidden;
        overflow-y: auto;
        margin: 0.35rem 0 0.8rem 0;
        background: #FFFFFF;
        max-height: 1100px;
    }
    .compact-wide-table-wrap table {
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
        font-size: 0.74rem;
        line-height: 1.12;
    }
    .compact-wide-table-wrap th {
        background: #F1F5F9;
        color: #0F172A;
        font-weight: 700;
        padding: 5px 4px;
        border-bottom: 1px solid #D8DEE9;
        text-align: center;
        white-space: normal;
        overflow-wrap: anywhere;
        vertical-align: bottom;
    }
    .compact-wide-table-wrap td {
        padding: 4px 4px;
        border-bottom: 1px solid #EEF2F7;
        vertical-align: middle;
        overflow-wrap: anywhere;
    }
    .compact-wide-table-wrap td.num {
        text-align: right;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Finanční analýza pro vodohospodářskou investici</div>', unsafe_allow_html=True)

INPUT_CATEGORIES: Dict[str, list[str]] = {
    "Projekt a časový horizont": ["capex_year", "revenue_start_year", "model_end_year"],
    "Investice a DCF": ["capex_amount", "discount_rate", "exit_discount_rate", "income_method_horizon_years"],
    "Voda, kapacita a výnosy": [
        "water_price_start",
        "water_price_escalation",
        "total_area_m2",
        "total_consumption_m3_day",
        "potable_water_share",
        "operating_days",
    ],
    "OPEX předpoklady": ["opex_monthly", "opex_escalation"],
    "Obnova technologie": ["renewal_interval_years", "renewal_capex_pct"],
    "Exit a ocenění": [
        "residual_life_years",
        "ebitda_multiple_low",
        "ebitda_multiple_base",
        "ebitda_multiple_high",
    ],
}
KEY_TO_CATEGORY = {key: category for category, keys in INPUT_CATEGORIES.items() for key in keys}


def _default_assumptions_df() -> pd.DataFrame:
    d = asdict(ModelInputs())
    rows = [
        ["capex_amount", "CAPEX", d["capex_amount"], "číslo", "Kč", "Investice do systému"],
        ["capex_year", "Rok investice", d["capex_year"], "rok", "rok", "Rok, kdy proběhne investice"],
        ["revenue_start_year", "Začátek výnosů", d["revenue_start_year"], "rok", "rok", "První rok výnosů"],
        ["model_end_year", "Konec modelu", d["model_end_year"], "rok", "rok", "Poslední rok DCF modelu"],
        ["discount_rate", "Diskontní sazba", d["discount_rate"], "procento", "desetinně", "5,5 % zadejte jako 0,055"],
        ["exit_discount_rate", "Diskontní sazba výnosové metody", d["exit_discount_rate"], "procento", "desetinně", "Požadovaný výnos kupujícího; 5,5 % zadejte jako 0,055"],
        ["income_method_horizon_years", "Oceňovací horizont výnosové metody", d["income_method_horizon_years"], "číslo", "let", "Horní limit let budoucích Free CF; výpočet nepřekročí konec modelové životnosti"],
        ["water_price_start", "Cena pitné vody – start", d["water_price_start"], "číslo", "Kč/m³", "Startovací cena bez DPH"],
        ["water_price_escalation", "Roční inflace ceny vody", d["water_price_escalation"], "procento", "desetinně", "2,5 % zadejte jako 0,025"],
        ["total_area_m2", "Celková plocha hal", d["total_area_m2"], "číslo", "m²", "Souhrnná plocha areálu"],
        ["total_consumption_m3_day", "Celková spotřeba areálu", d["total_consumption_m3_day"], "číslo", "m³/den", "Při plném obsazení"],
        ["potable_water_share", "Podíl pitné vody", d["potable_water_share"], "procento", "desetinně", "100 % zadejte jako 1,00"],
        ["operating_days", "Provozní dny", d["operating_days"], "číslo", "dní/rok", "Typicky 365"],
        ["opex_monthly", "Fixní OPEX měsíčně", d["opex_monthly"], "číslo", "Kč/měs.", "Můžete nahradit součtem OPEX detailu"],
        ["opex_escalation", "Roční inflace OPEX", d["opex_escalation"], "procento", "desetinně", "2,5 % zadejte jako 0,025"],
        ["renewal_interval_years", "Interval obnovy technologie", d["renewal_interval_years"], "číslo", "let", "Pracovní předpoklad: pravidelná obnova části technologie"],
        ["renewal_capex_pct", "Obnova technologie", d["renewal_capex_pct"], "procento", "% z CAPEX", "Rozpočítává se rovnoměrně do OPEX jako roční rezerva"],
        ["residual_life_years", "Životnost pro zůstatkovou hodnotu", d["residual_life_years"], "číslo", "let", "Pro zůstatkovou metodu"],
        ["ebitda_multiple_low", "EV/EBITDA low", d["ebitda_multiple_low"], "multiple", "x", "Pesimistický scénář"],
        ["ebitda_multiple_base", "EV/EBITDA base", d["ebitda_multiple_base"], "multiple", "x", "Base case scénář"],
        ["ebitda_multiple_high", "EV/EBITDA high", d["ebitda_multiple_high"], "multiple", "x", "Optimistický scénář"],
    ]
    df = pd.DataFrame(rows, columns=["Klíč", "Vstup", "Hodnota", "Typ", "Jednotka", "Poznámka"])
    # Pandas 3 is strict about assigning text into a numeric column.
    # Keeping Hodnota as object lets the user edit values in text inputs without TypeError.
    df["Hodnota"] = df["Hodnota"].astype("object")
    df.insert(1, "Kategorie", df["Klíč"].map(KEY_TO_CATEGORY).fillna("Ostatní"))
    return df

def _normalize_assumptions_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    default = _default_assumptions_df()
    for col in default.columns:
        if col not in out.columns:
            if col == "Kategorie":
                out[col] = out.get("Klíč", pd.Series(dtype=str)).map(KEY_TO_CATEGORY).fillna("Ostatní")
            else:
                out[col] = ""
    values = dict(zip(out["Klíč"].astype(str), out["Hodnota"])) if "Klíč" in out.columns else {}
    notes = dict(zip(out["Klíč"].astype(str), out["Poznámka"])) if "Klíč" in out.columns and "Poznámka" in out.columns else {}
    normalized = default.copy()
    normalized["Hodnota"] = normalized.apply(lambda r: values.get(str(r["Klíč"]), r["Hodnota"]), axis=1).astype("object")
    normalized["Poznámka"] = normalized.apply(lambda r: notes.get(str(r["Klíč"]), r["Poznámka"]), axis=1)
    return normalized


def _hall_sort_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.fillna("").astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(9999)


def _normalize_halls_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Hala", "Plocha_m2", "Rok_spusteni"]:
        if col not in out.columns:
            out[col] = "" if col == "Hala" else 0
    out = out[["Hala", "Plocha_m2", "Rok_spusteni"]].copy()
    out["Hala"] = out["Hala"].fillna("").astype(str)
    out["Plocha_m2"] = pd.to_numeric(out["Plocha_m2"], errors="coerce").fillna(0.0)
    out["Rok_spusteni"] = pd.to_numeric(out["Rok_spusteni"], errors="coerce").fillna(0).astype(int)
    filled = out["Hala"].str.strip().ne("") | out["Plocha_m2"].ne(0) | out["Rok_spusteni"].ne(0)
    out = out[filled].copy()
    out = out.assign(_hall_number=_hall_sort_number(out["Hala"]))
    return out.sort_values(["_hall_number", "Hala"], ascending=[True, True]).drop(columns=["_hall_number"]).reset_index(drop=True)

def _normalize_opex_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Typ / poznámka" not in out.columns:
        typ = out["Typ"].fillna("").astype(str) if "Typ" in out.columns else ""
        poz = out["Poznamka"].fillna("").astype(str) if "Poznamka" in out.columns else ""
        if hasattr(typ, "str"):
            combined = typ.str.strip()
            if hasattr(poz, "str"):
                combined = combined.where(poz.str.strip().eq(""), combined + " – " + poz.str.strip())
            out["Typ / poznámka"] = combined.str.strip(" –")
        else:
            out["Typ / poznámka"] = ""
    for col in ["Oblast", "Mesicne_Kc", "Typ / poznámka"]:
        if col not in out.columns:
            out[col] = "" if col != "Mesicne_Kc" else 0
    out = out[["Oblast", "Mesicne_Kc", "Typ / poznámka"]].copy()
    out["Oblast"] = out["Oblast"].fillna("").astype(str)
    out["Typ / poznámka"] = out["Typ / poznámka"].fillna("").astype(str)
    out["Mesicne_Kc"] = pd.to_numeric(out["Mesicne_Kc"], errors="coerce").fillna(0.0)
    filled = out["Oblast"].str.strip().ne("") | out["Typ / poznámka"].str.strip().ne("") | out["Mesicne_Kc"].ne(0)
    return out[filled].reset_index(drop=True)




def _format_editor_value(value: Any) -> str:
    """Return a safe text value for Streamlit data_editor.

    Streamlit infers a numeric dtype for all-number subsets. Because the
    Hodnota column is intentionally configured as a TextColumn (it contains
    both text and numbers across categories), each subset must be converted
    to string before editing. This prevents StreamlitAPIException about
    TextColumn being incompatible with FLOAT data.
    """
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _prepare_assumption_subset_for_editor(subset: pd.DataFrame) -> pd.DataFrame:
    out = subset.copy()
    out["Hodnota"] = out["Hodnota"].map(_format_editor_value).astype(str)
    return out


def _parse_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"Chybí hodnota pro {field_name}.")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\xa0", "").replace(" ", "").replace("%", "").replace(",", ".")
    if text == "":
        raise ValueError(f"Chybí hodnota pro {field_name}.")
    return float(text)


def _parse_int(value: Any, field_name: str) -> int:
    return int(round(_parse_float(value, field_name)))


def _inputs_from_assumptions(df: pd.DataFrame) -> ModelInputs:
    values: Dict[str, Any] = dict(zip(df["Klíč"].astype(str), df["Hodnota"]))
    return ModelInputs(
        project_name=str(values.get("project_name", "")).strip() or ModelInputs().project_name,
        currency=str(values.get("currency", "Kč")).strip() or "Kč",
        capex_amount=_parse_float(values.get("capex_amount"), "CAPEX"),
        capex_year=_parse_int(values.get("capex_year"), "Rok investice"),
        revenue_start_year=_parse_int(values.get("revenue_start_year"), "Začátek výnosů"),
        model_end_year=_parse_int(values.get("model_end_year"), "Konec modelu"),
        discount_rate=_parse_float(values.get("discount_rate"), "Diskontní sazba"),
        exit_discount_rate=_parse_float(values.get("exit_discount_rate"), "Diskontní sazba výnosové metody"),
        income_method_horizon_years=_parse_int(values.get("income_method_horizon_years", ModelInputs().income_method_horizon_years), "Oceňovací horizont výnosové metody"),
        water_price_start=_parse_float(values.get("water_price_start"), "Cena pitné vody"),
        water_price_escalation=_parse_float(values.get("water_price_escalation"), "Roční inflace ceny vody"),
        opex_monthly=_parse_float(values.get("opex_monthly"), "Fixní OPEX měsíčně"),
        opex_escalation=_parse_float(values.get("opex_escalation"), "Roční inflace OPEX"),
        renewal_interval_years=_parse_int(values.get("renewal_interval_years"), "Interval obnovy technologie"),
        renewal_capex_pct=_parse_float(values.get("renewal_capex_pct"), "Obnova technologie"),
        terminal_growth_rate=_parse_float(values.get("terminal_growth_rate", ModelInputs().terminal_growth_rate), "Obnova technologie"),
        total_area_m2=_parse_float(values.get("total_area_m2"), "Celková plocha hal"),
        total_consumption_m3_day=_parse_float(values.get("total_consumption_m3_day"), "Celková spotřeba areálu"),
        potable_water_share=_parse_float(values.get("potable_water_share"), "Podíl pitné vody"),
        operating_days=_parse_int(values.get("operating_days"), "Provozní dny"),
        residual_life_years=_parse_int(values.get("residual_life_years"), "Životnost"),
        ebitda_multiple_low=_parse_float(values.get("ebitda_multiple_low"), "EV/EBITDA low"),
        ebitda_multiple_base=_parse_float(values.get("ebitda_multiple_base"), "EV/EBITDA base"),
        ebitda_multiple_high=_parse_float(values.get("ebitda_multiple_high"), "EV/EBITDA high"),
    )


def _df_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    left = a.fillna("").reset_index(drop=True).astype(str)
    right = b.fillna("").reset_index(drop=True).astype(str)
    if list(left.columns) != list(right.columns):
        return False
    return left.equals(right)


def _snapshot() -> dict[str, Any]:
    return {
        "assumptions": st.session_state.assumptions_df.to_dict(orient="records"),
        "halls": st.session_state.halls_df.to_dict(orient="records"),
        "opex": st.session_state.opex_df.to_dict(orient="records"),
        "use_opex_detail": bool(st.session_state.use_opex_detail),
    }


def _restore_snapshot(snapshot: dict[str, Any]) -> None:
    st.session_state.assumptions_df = _normalize_assumptions_df(pd.DataFrame(snapshot.get("assumptions", [])))
    st.session_state.halls_df = _normalize_halls_df(pd.DataFrame(snapshot.get("halls", DEFAULT_HALLS.to_dict(orient="records"))))
    st.session_state.opex_df = _normalize_opex_df(pd.DataFrame(snapshot.get("opex", DEFAULT_OPEX.to_dict(orient="records"))))
    st.session_state.use_opex_detail = bool(snapshot.get("use_opex_detail", False))
    st.session_state.editor_version += 1


def _snapshot_signature(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str)


def _push_history() -> None:
    current = _snapshot()
    history: list[dict[str, Any]] = st.session_state.history
    if not history or _snapshot_signature(history[-1]) != _snapshot_signature(current):
        history.append(current)
        st.session_state.history = history[-25:]


def _reset_editor_widgets() -> None:
    st.session_state.editor_version += 1


def _reset_all() -> None:
    _push_history()
    st.session_state.assumptions_df = _default_assumptions_df()
    st.session_state.halls_df = _normalize_halls_df(DEFAULT_HALLS.copy())
    st.session_state.opex_df = _normalize_opex_df(DEFAULT_OPEX.copy())
    st.session_state.use_opex_detail = False
    st.session_state.last_changed_input_key = None
    _reset_editor_widgets()


def _reset_one_input(input_key: str) -> None:
    default = _default_assumptions_df()
    if input_key not in set(default["Klíč"]):
        return
    _push_history()
    default_value = default.loc[default["Klíč"] == input_key, "Hodnota"].iloc[0]
    st.session_state.assumptions_df.loc[st.session_state.assumptions_df["Klíč"] == input_key, "Hodnota"] = default_value
    st.session_state.last_changed_input_key = None
    _reset_editor_widgets()


def _reset_selected_or_last_input() -> bool:
    input_key = st.session_state.get("last_changed_input_key")
    if not input_key:
        return False
    _reset_one_input(str(input_key))
    return True


def _undo_last_change() -> None:
    history: list[dict[str, Any]] = st.session_state.history
    if history:
        snapshot = history.pop()
        st.session_state.history = history
        _restore_snapshot(snapshot)


def _load_scenario(payload: dict[str, Any]) -> None:
    _push_history()
    if "assumptions" in payload:
        assumptions_payload = payload.get("assumptions", [])
    else:
        # Backward compatibility: older sample JSONs stored ModelInputs as a flat dictionary.
        default = _default_assumptions_df()
        assumptions_payload = default.copy()
        for key, value in payload.items():
            if key in set(assumptions_payload["Klíč"]):
                assumptions_payload.loc[assumptions_payload["Klíč"] == key, "Hodnota"] = value
        assumptions_payload = assumptions_payload.to_dict(orient="records")
    st.session_state.assumptions_df = _normalize_assumptions_df(pd.DataFrame(assumptions_payload))
    st.session_state.halls_df = _normalize_halls_df(pd.DataFrame(payload.get("halls", DEFAULT_HALLS.to_dict(orient="records"))))
    st.session_state.opex_df = _normalize_opex_df(pd.DataFrame(payload.get("opex", DEFAULT_OPEX.to_dict(orient="records"))))
    st.session_state.use_opex_detail = bool(payload.get("use_opex_detail", False))
    _reset_editor_widgets()


def _compact_table(df: pd.DataFrame, height: int = 520) -> None:
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def _format_money_table(df: pd.DataFrame, height: int = 650) -> None:
    st.dataframe(df.style.format(precision=0, thousands=" "), use_container_width=True, hide_index=True, height=height)


def _fmt_int_value(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
        return f"{int(round(float(value)))}"
    except Exception:
        return str(value)


def _fmt_money_value(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
        return f"{float(value):,.0f}".replace(",", " ")
    except Exception:
        return str(value)


def _annual_renewal_reserve(inputs: ModelInputs) -> float:
    try:
        interval = int(inputs.renewal_interval_years)
        pct = float(inputs.renewal_capex_pct)
        capex = float(inputs.capex_amount)
        if interval <= 0 or pct <= 0 or capex <= 0:
            return 0.0
        return capex * pct / interval
    except Exception:
        return 0.0


def _monthly_renewal_reserve(inputs: ModelInputs) -> float:
    return _annual_renewal_reserve(inputs) / 12.0


def _fmt_multiple_label(value: Any) -> str:
    try:
        x = float(value)
        if x.is_integer():
            return str(int(x))
        return (f"{x:.1f}".rstrip("0").rstrip(".")).replace(".", ",")
    except Exception:
        return str(value)




def _fmt_percent_value(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "–"
        return f"{float(value):.1%}".replace(".", ",")
    except Exception:
        return str(value)


def _html_escape(value: Any) -> str:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _render_html_table(
    df: pd.DataFrame,
    wrapper_class: str,
    money_cols: list[str] | None = None,
    percent_cols: list[str] | None = None,
    int_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
) -> None:
    money_cols = money_cols or []
    percent_cols = percent_cols or []
    int_cols = int_cols or []
    text_cols = text_cols or []
    if df is None or df.empty:
        st.info("Tabulka nemá žádné řádky k zobrazení.")
        return
    cols = list(df.columns)
    header = "".join(f"<th>{_html_escape(c)}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cls = ""
            if c in money_cols:
                val = _fmt_money_value(v)
                cls = " class='num'"
            elif c in percent_cols:
                val = _fmt_percent_value(v)
                cls = " class='num'"
            elif c in int_cols:
                val = _fmt_int_value(v)
                cls = " class='num'"
            else:
                val = v if c in text_cols else v
            cells.append(f"<td{cls}>{_html_escape(val)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f"<div class='pretty-table-wrap {wrapper_class}'><table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _pretty_table(
    df: pd.DataFrame,
    money_cols: list[str] | None = None,
    percent_cols: list[str] | None = None,
    int_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
) -> None:
    _render_html_table(df, "", money_cols, percent_cols, int_cols, text_cols)


def _compact_pretty_table(
    df: pd.DataFrame,
    money_cols: list[str] | None = None,
    percent_cols: list[str] | None = None,
    int_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
) -> None:
    _render_html_table(df, "compact-pretty-table-wrap", money_cols, percent_cols, int_cols, text_cols)



def _compact_wide_table(
    df: pd.DataFrame,
    money_cols: list[str] | None = None,
    percent_cols: list[str] | None = None,
    int_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
) -> None:
    money_cols = money_cols or []
    percent_cols = percent_cols or []
    int_cols = int_cols or []
    text_cols = text_cols or []
    if df is None or df.empty:
        st.info("Tabulka nemá žádné řádky k zobrazení.")
        return
    cols = list(df.columns)
    header = "".join(f"<th>{_html_escape(c)}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cls = ""
            if c in money_cols:
                val = _fmt_money_value(v)
                cls = " class='num'"
            elif c in percent_cols:
                val = _fmt_percent_value(v)
                cls = " class='num'"
            elif c in int_cols:
                val = _fmt_int_value(v)
                cls = " class='num'"
            else:
                val = v
            cells.append(f"<td{cls}>{_html_escape(val)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        "<div class='compact-wide-table-wrap'><table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _risk_analysis_table(inputs: ModelInputs, halls_df: pd.DataFrame, opex_df: pd.DataFrame) -> pd.DataFrame:
    base_model = calculate_model(inputs, halls_df, opex_df)
    base_metrics: Dict[str, Any] = base_model["metrics"]  # type: ignore[assignment]
    base_npv = float(base_metrics.get("NPV - čistá současná hodnota", 0.0) or 0.0)

    scenarios: list[tuple[str, str, ModelInputs]] = [
        ("Základní scénář", "Aktuální vstupy bez změny", replace(inputs)),
        ("Nižší cena vody", "Cena vody je o 10 % nižší", replace(inputs, water_price_start=inputs.water_price_start * 0.90)),
        (
            "Nižší odběrové množství",
            "Odběrové množství je o 15 % nižší",
            replace(inputs, total_consumption_m3_day=inputs.total_consumption_m3_day * 0.85),
        ),
        ("Vyšší OPEX", "Provozní náklady jsou o 15 % vyšší", replace(inputs, opex_monthly=inputs.opex_monthly * 1.15)),
        ("Vyšší CAPEX", "Počáteční investice je o 15 % vyšší", replace(inputs, capex_amount=inputs.capex_amount * 1.15)),
        ("Vyšší WACC", "Diskontní sazba je vyšší o 1 procentní bod", replace(inputs, discount_rate=inputs.discount_rate + 0.01)),
        (
            "Kombinovaný downside",
            "Cena vody -10 %, odběrové množství -15 %, OPEX +15 %, CAPEX +15 %, WACC +1 p. b.",
            replace(
                inputs,
                water_price_start=inputs.water_price_start * 0.90,
                total_consumption_m3_day=inputs.total_consumption_m3_day * 0.85,
                opex_monthly=inputs.opex_monthly * 1.15,
                capex_amount=inputs.capex_amount * 1.15,
                discount_rate=inputs.discount_rate + 0.01,
            ),
        ),
    ]
    rows = []
    for name, change, scenario_inputs in scenarios:
        scenario_model = calculate_model(scenario_inputs, halls_df, opex_df)
        m: Dict[str, Any] = scenario_model["metrics"]  # type: ignore[assignment]
        npv = float(m.get("NPV - čistá současná hodnota", 0.0) or 0.0)
        rows.append(
            {
                "Scénář": name,
                "Co se mění": change,
                "NPV": npv,
                "Změna NPV oproti základnímu scénáři": npv - base_npv,
                "IRR": m.get("IRR"),
                "Doba návratnosti": _years_label(m.get("Diskontovaná návratnost - počet let")),
                "Čistý výnos": m.get("Čistý výnos"),
            }
        )
    return pd.DataFrame(rows)

def _exit_display_table(model: Dict[str, Any]) -> pd.DataFrame:
    inputs: ModelInputs = model["inputs"]  # type: ignore[assignment]
    exit_df: pd.DataFrame = model["exit_analysis"].copy()  # type: ignore[assignment]

    base_multiple = _fmt_multiple_label(inputs.ebitda_multiple_base)
    low_multiple = _fmt_multiple_label(inputs.ebitda_multiple_low)
    high_multiple = _fmt_multiple_label(inputs.ebitda_multiple_high)

    source_cols = {
        f"Cena EBITDA {base_multiple}x": f"Cena dle {inputs.ebitda_multiple_base:.1f}x EBITDA (Kč)",
        f"Cena EBITDA {low_multiple}x": f"Cena dle {inputs.ebitda_multiple_low:.1f}x EBITDA (Kč)",
        f"Cena EBITDA {high_multiple}x": f"Cena dle {inputs.ebitda_multiple_high:.1f}x EBITDA (Kč)",
    }

    out = pd.DataFrame(
        {
            "Rok provozu": exit_df["Rok provozu"],
            "Kal. rok": exit_df["Kalendářní rok"],
            "Free CF": exit_df["FCF v roce (Kč)"],
            "Zůstatková cena": exit_df["Zůstatková hodnota (Kč)"],
            "Výnosová cena": exit_df["Prodej za výnosovou metodu (Kč)"],
            "EBITDA": exit_df["EBITDA v roce (Kč)"],
        }
    )
    # Pořadí: base, low, high — tedy ve výchozím nastavení 11, 8, 14.
    for display_col, source_col in source_cols.items():
        if source_col in exit_df.columns:
            out[display_col] = exit_df[source_col]
        else:
            out[display_col] = 0.0
    return out


def _show_exit_table(exit_display: pd.DataFrame, height: int = 850) -> None:
    int_cols = ["Rok provozu", "Kal. rok"]
    money_cols = [c for c in exit_display.columns if c not in int_cols]
    formatters = {col: _fmt_int_value for col in int_cols}
    formatters.update({col: _fmt_money_value for col in money_cols})
    st.dataframe(exit_display.style.format(formatters), use_container_width=True, hide_index=True, height=height)


def _show_exit_explanations(inputs: ModelInputs, compact: bool = False) -> None:
    base = _fmt_multiple_label(inputs.ebitda_multiple_base)
    low = _fmt_multiple_label(inputs.ebitda_multiple_low)
    high = _fmt_multiple_label(inputs.ebitda_multiple_high)
    if compact:
        return

    st.markdown(
        f"""
        **Jednoduché vysvětlení tabulky**

        Tabulka ukazuje, jak by projekt vypadal, kdyby se prodal v různých letech provozu. Každý řádek je jeden možný rok prodeje. Investor tak vidí, jestli je lepší projekt držet déle, nebo ho prodat dříve.

        **Co znamenají metody ocenění**

        - **Free CF** znamená roční volný peněžní tok. Laicky: kolik peněz projekt v daném roce přinese po provozních nákladech.
        - **Zůstatková cena** vychází z původní investice a zbývající životnosti. Laicky: jaká část investice ještě není „spotřebovaná“. Hodí se jako opatrnější technický pohled na hodnotu.
        - **Výnosová cena** oceňuje projekt podle budoucích Free CF, které může kupující získat po koupi. V modelu se počítají pouze roky do konce nastavené životnosti investice. Proto je v posledním roce hodnota 0 – po tomto roce už model další cash flow neoceňuje, i když projekt může technicky fungovat dál.
        - **Cena EBITDA** je ocenění podle provozního výsledku. Vezme se EBITDA za daný rok a vynásobí se vybraným násobkem. Tuto metodu investoři používají hlavně pro rychlé porovnání s podobnými projekty.

        **Co znamenají násobky {low}x, {base}x a {high}x**

        Násobek není pevné pravidlo, ale scénář ocenění. Čím je projekt jistější, stabilnější a atraktivnější, tím vyšší násobek může investor připustit. Čím je riziko vyšší, tím nižší násobek dává smysl.

        - **{low}x EBITDA** je opatrná varianta. Používá se, když je projekt rizikovější, výnosy ještě nejsou stabilní, náklady mohou růst, nebo investor chce větší bezpečnostní rezervu.
        - **{base}x EBITDA** je základní varianta. Používá se jako hlavní pracovní scénář, když projekt funguje podle očekávání a riziko je přiměřené.
        - **{high}x EBITDA** je optimističtější varianta. Používá se, když je projekt stabilní, má dobře předvídatelné výnosy, nízké provozní riziko a pro kupujícího je strategicky zajímavý.

        """
    )

def _merge_assumption_edits(original: pd.DataFrame, edited_subset: pd.DataFrame) -> pd.DataFrame:
    updated = original.copy()
    for _, row in edited_subset.iterrows():
        key = str(row["Klíč"])
        updated.loc[updated["Klíč"].astype(str) == key, "Hodnota"] = row["Hodnota"]
    return updated


def _scenario_payload() -> dict[str, Any]:
    return {
        "assumptions": st.session_state.assumptions_df.to_dict(orient="records"),
        "halls": st.session_state.halls_df.to_dict(orient="records"),
        "opex": st.session_state.opex_df.to_dict(orient="records"),
        "use_opex_detail": bool(st.session_state.use_opex_detail),
    }


# ------------------------- Session state -------------------------
if "assumptions_df" not in st.session_state:
    st.session_state.assumptions_df = _default_assumptions_df()
else:
    st.session_state.assumptions_df = _normalize_assumptions_df(st.session_state.assumptions_df)
if "halls_df" not in st.session_state:
    st.session_state.halls_df = _normalize_halls_df(DEFAULT_HALLS.copy())
else:
    st.session_state.halls_df = _normalize_halls_df(st.session_state.halls_df)
if "opex_df" not in st.session_state:
    st.session_state.opex_df = _normalize_opex_df(DEFAULT_OPEX.copy())
else:
    st.session_state.opex_df = _normalize_opex_df(st.session_state.opex_df)
if "use_opex_detail" not in st.session_state:
    st.session_state.use_opex_detail = True
if "history" not in st.session_state:
    st.session_state.history = []
if "editor_version" not in st.session_state:
    st.session_state.editor_version = 0
if "last_changed_input_key" not in st.session_state:
    st.session_state.last_changed_input_key = None


with st.sidebar:
    st.header("Rychlé ovládání")

    undo_disabled = len(st.session_state.history) == 0
    if st.button("↩️ Vrátit poslední změnu", use_container_width=True, disabled=undo_disabled):
        _undo_last_change()
        st.rerun()

    last_key = st.session_state.get("last_changed_input_key")
    if st.button("Resetovat poslední upravený vstup", use_container_width=True, disabled=not bool(last_key)):
        if _reset_selected_or_last_input():
            st.rerun()

    if st.button("🔄 Resetovat všechny vstupy", use_container_width=True):
        _reset_all()
        st.rerun()

    st.divider()
    reset_options = st.session_state.assumptions_df[["Klíč", "Vstup"]].copy()
    selected_label = st.selectbox("Resetovat konkrétní vstup", reset_options["Vstup"].tolist())
    selected_key = reset_options.loc[reset_options["Vstup"] == selected_label, "Klíč"].iloc[0]
    if st.button("Resetovat vybraný vstup", use_container_width=True):
        _reset_one_input(str(selected_key))
        st.rerun()

    st.divider()
    st.write("Scénáře")
    st.download_button(
        "💾 Uložit scénář jako JSON",
        data=json.dumps(_scenario_payload(), ensure_ascii=False, indent=2),
        file_name="financial_model_scenario.json",
        mime="application/json",
        use_container_width=True,
    )
    uploaded_scenario = st.file_uploader("Načíst uložený scénář", type=["json"])
    if uploaded_scenario is not None and st.button("Načíst scénář", use_container_width=True):
        try:
            payload = json.load(uploaded_scenario)
            _load_scenario(payload)
            st.success("Scénář byl načten.")
            st.rerun()
        except Exception as exc:
            st.error(f"Scénář se nepodařilo načíst: {exc}")


tab_inputs, tab_dashboard, tab_dcf, tab_exit, tab_cf_sale, tab_sensitivity, tab_export = st.tabs(
    ["Vstupy", "Dashboard", "DCF", "Exit & EBITDA", "CF + prodej", "Analýza rizik", "Export"]
)

with tab_inputs:
    st.subheader("1. Základní vstupy modelu")

    for category in INPUT_CATEGORIES.keys():
        with st.container(border=True):
            st.markdown(f'<div class="input-topic-title">{category}</div>', unsafe_allow_html=True)
            subset = st.session_state.assumptions_df.loc[
                st.session_state.assumptions_df["Kategorie"] == category,
                ["Klíč", "Vstup", "Hodnota", "Jednotka", "Poznámka"],
            ].copy()

            h1, h2, h3, h4, h5 = st.columns([2.0, 1.25, 0.85, 2.6, 0.35])
            h1.caption("Vstup")
            h2.caption("Hodnota")
            h3.caption("Jednotka")
            h4.caption("Poznámka")
            h5.caption(" ")

            for _, row in subset.iterrows():
                input_key = str(row["Klíč"])
                value_current = _format_editor_value(row["Hodnota"])
                note_current = "" if pd.isna(row["Poznámka"]) else str(row["Poznámka"])
                unit_current = "" if pd.isna(row["Jednotka"]) else str(row["Jednotka"])

                c1, c2, c3, c4, c5 = st.columns([2.0, 1.25, 0.85, 2.6, 0.35])
                c1.markdown(str(row["Vstup"]))
                value_widget_key = f"ass_value_{input_key}_{st.session_state.editor_version}"
                note_widget_key = f"ass_note_{input_key}_{st.session_state.editor_version}"
                value_new = c2.text_input(
                    "Hodnota",
                    value=value_current,
                    key=value_widget_key,
                    label_visibility="collapsed",
                )
                c3.markdown(unit_current if unit_current else "—")
                note_new = c4.text_input(
                    "Poznámka",
                    value=note_current,
                    key=note_widget_key,
                    label_visibility="collapsed",
                )
                if c5.button("↺", key=f"reset_value_{input_key}_{st.session_state.editor_version}", help="Vrátit hodnotu na původní"):
                    _reset_one_input(input_key)
                    st.rerun()

                mask = st.session_state.assumptions_df["Klíč"].astype(str) == input_key
                if str(value_new) != value_current:
                    _push_history()
                    st.session_state.assumptions_df.loc[mask, "Hodnota"] = value_new
                    st.session_state.last_changed_input_key = input_key
                if str(note_new) != note_current:
                    _push_history()
                    st.session_state.assumptions_df.loc[mask, "Poznámka"] = note_new
                    st.session_state.last_changed_input_key = input_key

    st.divider()
    st.subheader("2. Náběh hal")
    halls_before = _normalize_halls_df(st.session_state.halls_df.copy())
    halls_rows: list[dict[str, Any]] = []
    halls_changed = False
    remove_hall_idx: int | None = None

    with st.container(border=True):
        st.markdown('<div class="table-header">Haly v modelu</div>', unsafe_allow_html=True)
        h1, h2, h3, h4 = st.columns([1.6, 1.5, 1.1, 0.55])
        h1.caption("Hala")
        h2.caption("Plocha (m²)")
        h3.caption("Rok spuštění")
        h4.caption(" ")

        for idx, row in halls_before.iterrows():
            c1, c2, c3, c4 = st.columns([1.6, 1.5, 1.1, 0.55])
            hall_name_current = str(row["Hala"])
            area_current = float(row["Plocha_m2"])
            year_current = int(row["Rok_spusteni"])

            hall_name_new = c1.text_input(
                "Hala",
                value=hall_name_current,
                key=f"hall_name_{idx}_{st.session_state.editor_version}",
                label_visibility="collapsed",
            )
            area_new_text = c2.text_input(
                "Plocha (m²)",
                value=_fmt_money_value(area_current),
                key=f"hall_area_{idx}_{st.session_state.editor_version}",
                label_visibility="collapsed",
            )
            year_new = c3.number_input(
                "Rok spuštění",
                min_value=2020,
                max_value=2150,
                step=1,
                value=year_current,
                key=f"hall_year_{idx}_{st.session_state.editor_version}",
                label_visibility="collapsed",
            )
            if c4.button("−", key=f"remove_hall_row_{idx}_{st.session_state.editor_version}", help="Odebrat řádek", use_container_width=True):
                remove_hall_idx = idx

            try:
                area_new = _parse_float(area_new_text, "Plocha haly")
                halls_rows.append({"Hala": str(hall_name_new).strip(), "Plocha_m2": float(area_new), "Rok_spusteni": int(year_new)})
                if str(hall_name_new).strip() != hall_name_current or float(area_new) != area_current or int(year_new) != year_current:
                    halls_changed = True
            except Exception:
                halls_rows.append({"Hala": hall_name_current, "Plocha_m2": area_current, "Rok_spusteni": year_current})
                st.error(f"Plocha u řádku {idx + 1} musí být číslo. Původní hodnota zůstala zachována.")

    if remove_hall_idx is not None:
        _push_history()
        st.session_state.halls_df = _normalize_halls_df(halls_before.drop(halls_before.index[remove_hall_idx]).reset_index(drop=True))
        _reset_editor_widgets()
        st.rerun()

    halls_df = _normalize_halls_df(pd.DataFrame(halls_rows))
    if halls_changed and not _df_equal(halls_before, halls_df):
        _push_history()
        st.session_state.halls_df = halls_df
        _reset_editor_widgets()
        st.rerun()

    with st.expander("Přidat halu", expanded=False):
        ah1, ah2, ah3, ah4 = st.columns([1.6, 1.5, 1.1, 0.75])
        next_hall_number = len(st.session_state.halls_df) + 1
        new_hall_name = ah1.text_input(
            "Nová hala",
            value=f"Hala {next_hall_number}",
            key=f"new_hall_name_{st.session_state.editor_version}",
        )
        new_hall_area_text = ah2.text_input(
            "Plocha (m²)",
            value="0",
            key=f"new_hall_area_{st.session_state.editor_version}",
        )
        new_hall_year = ah3.number_input(
            "Rok spuštění",
            min_value=2020,
            max_value=2150,
            step=1,
            value=2028,
            key=f"new_hall_year_{st.session_state.editor_version}",
        )
        if ah4.button("Přidat", use_container_width=True, key=f"add_hall_button_{st.session_state.editor_version}"):
            try:
                new_hall_area = _parse_float(new_hall_area_text, "Plocha nové haly")
            except Exception:
                new_hall_area = 0.0
            if str(new_hall_name).strip() and float(new_hall_area) > 0:
                _push_history()
                new_row = pd.DataFrame([{"Hala": str(new_hall_name).strip(), "Plocha_m2": float(new_hall_area), "Rok_spusteni": int(new_hall_year)}])
                st.session_state.halls_df = _normalize_halls_df(pd.concat([st.session_state.halls_df, new_row], ignore_index=True))
                _reset_editor_widgets()
                st.rerun()
            else:
                st.warning("Vyplňte název haly a plochu větší než 0.")

    st.subheader("3. OPEX detail")
    opex_before = _normalize_opex_df(st.session_state.opex_df.copy())
    opex_rows: list[dict[str, Any]] = []
    opex_changed = False
    remove_opex_idx: int | None = None

    with st.container(border=True):
        st.markdown('<div class="table-header">Provozní náklady</div>', unsafe_allow_html=True)
        h1, h2, h3, h4 = st.columns([1.9, 1.35, 2.4, 0.55])
        h1.caption("Oblast")
        h2.caption("Měsíčně (Kč)")
        h3.caption("Typ / poznámka")
        h4.caption(" ")

        for idx, row in opex_before.iterrows():
            c1, c2, c3, c4 = st.columns([1.9, 1.35, 2.4, 0.55])
            oblast_current = str(row["Oblast"])
            amount_current = float(row["Mesicne_Kc"])
            note_current = str(row["Typ / poznámka"])

            oblast_new = c1.text_input(
                "Oblast",
                value=oblast_current,
                key=f"opex_area_{idx}_{st.session_state.editor_version}",
                label_visibility="collapsed",
            )
            amount_new_text = c2.text_input(
                "Měsíčně (Kč)",
                value=_fmt_money_value(amount_current),
                key=f"opex_amount_{idx}_{st.session_state.editor_version}",
                label_visibility="collapsed",
            )
            note_new = c3.text_input(
                "Typ / poznámka",
                value=note_current,
                key=f"opex_note_{idx}_{st.session_state.editor_version}",
                label_visibility="collapsed",
            )
            if c4.button("−", key=f"remove_opex_row_{idx}_{st.session_state.editor_version}", help="Odebrat řádek", use_container_width=True):
                remove_opex_idx = idx

            try:
                amount_new = _parse_float(amount_new_text, "OPEX měsíčně")
                opex_rows.append({"Oblast": str(oblast_new).strip(), "Mesicne_Kc": float(amount_new), "Typ / poznámka": str(note_new).strip()})
                if str(oblast_new).strip() != oblast_current or float(amount_new) != amount_current or str(note_new).strip() != note_current:
                    opex_changed = True
            except Exception:
                opex_rows.append({"Oblast": oblast_current, "Mesicne_Kc": amount_current, "Typ / poznámka": note_current})
                st.error(f"OPEX u řádku {idx + 1} musí být číslo. Původní hodnota zůstala zachována.")


    if remove_opex_idx is not None:
        _push_history()
        st.session_state.opex_df = _normalize_opex_df(opex_before.drop(opex_before.index[remove_opex_idx]).reset_index(drop=True))
        _reset_editor_widgets()
        st.rerun()

    opex_df = _normalize_opex_df(pd.DataFrame(opex_rows))
    if opex_changed and not _df_equal(opex_before, opex_df):
        _push_history()
        st.session_state.opex_df = opex_df
        _reset_editor_widgets()
        st.rerun()

    with st.expander("Přidat OPEX položku", expanded=False):
        ao1, ao2, ao3, ao4 = st.columns([1.9, 1.35, 2.4, 0.75])
        new_opex_area = ao1.text_input("Oblast", key=f"new_opex_area_{st.session_state.editor_version}")
        new_opex_amount_text = ao2.text_input("Měsíčně (Kč)", value="0", key=f"new_opex_amount_{st.session_state.editor_version}")
        new_opex_note = ao3.text_input("Typ / poznámka", key=f"new_opex_note_{st.session_state.editor_version}")
        if ao4.button("Přidat", use_container_width=True, key=f"add_opex_button_{st.session_state.editor_version}"):
            try:
                new_opex_amount = _parse_float(new_opex_amount_text, "Nový OPEX")
            except Exception:
                new_opex_amount = 0.0
            if str(new_opex_area).strip() and float(new_opex_amount) > 0:
                _push_history()
                new_row = pd.DataFrame([{"Oblast": str(new_opex_area).strip(), "Mesicne_Kc": float(new_opex_amount), "Typ / poznámka": str(new_opex_note).strip()}])
                st.session_state.opex_df = _normalize_opex_df(pd.concat([st.session_state.opex_df, new_row], ignore_index=True))
                _reset_editor_widgets()
                st.rerun()
            else:
                st.warning("Vyplňte oblast a částku větší než 0.")

    opex_sum = pd.to_numeric(st.session_state.opex_df["Mesicne_Kc"], errors="coerce").fillna(0).sum()
    cols_opex = st.columns(3)
    with cols_opex[0]:
        st.metric("OPEX celkem", f"{opex_sum:,.0f} Kč/měs.".replace(",", " "))
    with cols_opex[1]:
        st.metric("OPEX celkem", f"{opex_sum * 12:,.0f} Kč/rok".replace(",", " "))
    with cols_opex[2]:
        st.metric("Počet položek", f"{len(st.session_state.opex_df)}")

    use_opex_detail_value = st.checkbox(
        "Použít součet OPEX detailu jako OPEX měsíčně pro výpočet",
        value=bool(st.session_state.use_opex_detail),
        help="Když je zapnuto, model ignoruje hodnotu v řádku Fixní OPEX měsíčně a použije součet všech položek v OPEX detailu, včetně Obnovy technologie.",
        key=f"use_opex_detail_editor_{st.session_state.editor_version}",
    )
    if use_opex_detail_value != bool(st.session_state.use_opex_detail):
        _push_history()
        st.session_state.use_opex_detail = bool(use_opex_detail_value)

# Create model after input tab so changes are reflected across all tabs.
try:
    inputs = _inputs_from_assumptions(st.session_state.assumptions_df)
except Exception as exc:
    st.error(f"Některý vstup nejde přečíst: {exc}")
    st.stop()

if bool(st.session_state.use_opex_detail):
    inputs.opex_monthly = float(pd.to_numeric(st.session_state.opex_df["Mesicne_Kc"], errors="coerce").fillna(0).sum())
    inputs.renewal_included_in_opex = True

model = calculate_model(inputs, st.session_state.halls_df, st.session_state.opex_df)

with tab_dashboard:
    st.subheader("Dashboard")
    metrics_df = summarize_metrics(model)
    cols = st.columns(4)
    for i, row in metrics_df.head(4).iterrows():
        with cols[i % 4]:
            st.metric(row["Metrika"], row["Hodnota"])

    st.write("Přehled klíčových výstupů")
    _pretty_table(metrics_df, text_cols=["Metrika", "Hodnota"])

    st.markdown(
        """
        **NPV – čistá současná hodnota:** Ukazuje, kolik peněz (v dnešní hodnotě) projekt celkově vydělá nad rámec vložených prostředků. Kladná hodnota znamená, že projekt generuje hodnotu.

        **IRR:** Vnitřní výnosové procento. Laicky řečeno ukazuje, jaké průměrné roční zhodnocení projekt podle modelu přináší. Investor ho porovnává s požadovaným výnosem nebo s WACC. Pokud je IRR vyšší než WACC, projekt by měl z finančního pohledu dávat smysl.

        **WACC:** Průměrná požadovaná cena kapitálu. Laicky: minimální výnos, který by měl projekt vydělat, aby dával investorovi smysl vzhledem k riziku a ceně financování. V modelu se používá jako diskontní sazba pro převod budoucích peněz na dnešní hodnotu.
        """
    )

    st.subheader("Exitová analýza – přehled")
    exit_display_dashboard = _exit_display_table(model)
    exit_display_dashboard = exit_display_dashboard[exit_display_dashboard["Rok provozu"].isin([3, 5, 7, 10])].reset_index(drop=True)
    _pretty_table(
        exit_display_dashboard,
        money_cols=[c for c in exit_display_dashboard.columns if c not in ["Rok provozu", "Kal. rok"]],
        int_cols=["Rok provozu", "Kal. rok"],
    )

with tab_dcf:
    st.subheader("DCF výpočet")
    dcf_display = model["dcf"].copy()
    dcf_display = dcf_display.drop(columns=["Pitná voda (m³/rok)", "% plochy", "DF", "Obnova technologie (Kč)", "Obnova technologie v OPEX (Kč)"], errors="ignore")
    dcf_display = dcf_display.rename(
        columns={
            "Index": "Index",
            "Plocha v provozu (m²)": "Plocha v provozu (m²)",
            "Pitná voda (m³/den)": "Pitná voda (m³/den)",
            "Cena vody (Kč/m³)": "Cena vody (Kč/m³)",
            "Výnosy (Kč)": "Výnosy",
            "OPEX (Kč)": "OPEX vč. obnovy",
            "CAPEX (Kč)": "CAPEX",
            "FCF (Kč)": "FCF",
            "Měsíční FCF (Kč/měs.)": "Měsíční FCF",
            "PV FCF (Kč)": "PV FCF",
            "EBITDA (Kč)": "EBITDA",
        }
    )
    money_cols = [c for c in dcf_display.columns if c not in ["Rok", "Index"]]
    _compact_wide_table(dcf_display, money_cols=money_cols, int_cols=["Rok", "Index"])

with tab_exit:
    st.subheader("Exitová analýza")
    exit_display = _exit_display_table(model)
    _show_exit_table(exit_display, height=max(900, min(1400, 74 + 34 * (len(exit_display) + 2))))
    _show_exit_explanations(inputs, compact=False)

with tab_cf_sale:
    st.subheader("Cash flow + prodej po 5 a 10 letech")
    sale_income = model.get("sale_cashflow_income", pd.DataFrame())
    sale_ebitda = model.get("sale_cashflow_ebitda", pd.DataFrame())

    if isinstance(sale_income, pd.DataFrame) and not sale_income.empty:
        st.markdown("#### 1. Prodej výnosovou metodou")
        money_cols = [c for c in sale_income.columns if "(Kč)" in c]
        _compact_pretty_table(
            sale_income,
            money_cols=money_cols,
            int_cols=["Rok provozu", "Kal. rok prodeje"],
            text_cols=["Scénář"],
        )

        st.markdown("#### 2. Prodej metodou EBITDA 11x")
        money_cols = [c for c in sale_ebitda.columns if "(Kč)" in c]
        _compact_pretty_table(
            sale_ebitda,
            money_cols=money_cols,
            int_cols=["Rok provozu", "Kal. rok prodeje"],
            text_cols=["Scénář"],
        )
    else:
        st.info("Pro zadané období modelu nelze vytvořit scénář prodeje po 5 a 10 letech.")

with tab_sensitivity:
    st.subheader("Analýza rizik")
    risk_df = _risk_analysis_table(inputs, st.session_state.halls_df, st.session_state.opex_df)
    model["risk_analysis"] = risk_df.copy()
    st.markdown(
        "Analýza ukazuje, jak se změní základní ekonomika projektu, pokud nastane některé z hlavních rizik. "
        "Nejde o předpověď, ale o kontrolu citlivosti modelu na nepříznivé vstupy."
    )
    _pretty_table(
        risk_df,
        money_cols=["NPV", "Změna NPV oproti základnímu scénáři", "Čistý výnos"],
        percent_cols=["IRR"],
        text_cols=["Scénář", "Co se mění", "Doba návratnosti"],
    )

with tab_export:
    st.subheader("Export")
    st.write("Stáhněte si Excel s aktuálními vstupy a výstupy.")
    excel_bytes = create_excel_report(model)
    st.download_button(
        "Stáhnout investor-ready Excel",
        data=excel_bytes,
        file_name="investor_financial_analysis_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.download_button(
        "Stáhnout vstupy/scénář jako JSON",
        data=json.dumps(_scenario_payload(), ensure_ascii=False, indent=2),
        file_name="financial_model_inputs.json",
        mime="application/json",
    )

    st.warning("Model je automatický analytický výstup, ne investiční, daňové ani právní doporučení. Před zasláním investorům proveďte odbornou kontrolu vstupů a metodiky.")
