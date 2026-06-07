from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import numpy_financial as npf
except Exception:  # pragma: no cover - fallback for minimal installations
    npf = None


DEFAULT_HALLS = pd.DataFrame(
    [
        {"Hala": "Hala 1", "Plocha_m2": 17500, "Rok_spusteni": 2028},
        {"Hala": "Hala 2", "Plocha_m2": 6600, "Rok_spusteni": 2028},
        {"Hala": "Hala 3", "Plocha_m2": 27900, "Rok_spusteni": 2029},
        {"Hala": "Hala 4", "Plocha_m2": 30600, "Rok_spusteni": 2030},
        {"Hala": "Hala 5", "Plocha_m2": 34000, "Rok_spusteni": 2030},
        {"Hala": "Hala 6", "Plocha_m2": 42300, "Rok_spusteni": 2031},
        {"Hala": "Hala 7", "Plocha_m2": 49300, "Rok_spusteni": 2032},
    ]
)

DEFAULT_OPEX = pd.DataFrame(
    [
        {"Oblast": "Úprava vody – spotřební materiál", "Mesicne_Kc": 70000, "Typ / poznámka": "Spotřební materiál – sůl, chlornan, membrány"},
        {"Oblast": "Elektřina a pohon systému", "Mesicne_Kc": 25000, "Typ / poznámka": "Energie – čerpání, úpravna, AT stanice, MaR"},
        {"Oblast": "Servis a preventivní údržba technologie", "Mesicne_Kc": 25000, "Typ / poznámka": "Servis / údržba – pravidelný servis technologie"},
        {"Oblast": "Laboratorní rozbory a monitoring", "Mesicne_Kc": 15000, "Typ / poznámka": "Monitoring / compliance – rozbory a vzorkování"},
        {"Oblast": "Odborný zástupce / technický garant", "Mesicne_Kc": 25000, "Typ / poznámka": "Práce / odborná služba – dohled, komunikace s úřady"},
        {"Oblast": "Provozní obsluha a facility činnosti", "Mesicne_Kc": 20000, "Typ / poznámka": "Práce / provoz – kontrola technologie, odečty"},
        {"Oblast": "Administrace, fakturace, evidence", "Mesicne_Kc": 10000, "Typ / poznámka": "Práce / administrativa – fakturace, smluvní agenda"},
        {"Oblast": "Havarijní pohotovost a drobné opravy", "Mesicne_Kc": 10000, "Typ / poznámka": "Rezerva / servis – rezerva na zásahy a drobné opravy"},
        {"Oblast": "Obnova technologie", "Mesicne_Kc": 30000.0, "Typ / poznámka": "Rezerva na průběžnou obnovu technologie – částku lze upravit ručně"},
    ]
)

DEFAULT_SOURCES = pd.DataFrame(
    [
        {"Téma": "Plochy hal", "Zdroj": "Interní podklad", "URL / dokument": "uploaded file", "Poznámka": "Součet ploch hal v náběhové tabulce."},
        {"Téma": "Cena vody", "Zdroj": "Uživatel / vstup modelu", "URL / dokument": "doplnit", "Poznámka": "Cena pitné vody bez DPH, lze měnit v aplikaci."},
        {"Téma": "DCF metodika", "Zdroj": "Interní metodika modelu", "URL / dokument": "interní", "Poznámka": "NPV z ročních FCF diskontovaných k roku investice."},
        {"Téma": "EV/EBITDA", "Zdroj": "Vstup uživatele / comps", "URL / dokument": "doplnit", "Poznámka": "Multiple scénáře se nastavují ve vstupních předpokladech."},
    ]
)


@dataclass
class ModelInputs:
    project_name: str = "LPA Bavoryně – vodní systém"
    currency: str = "Kč"
    capex_amount: float = 20_000_000.0
    capex_year: int = 2027
    revenue_start_year: int = 2028
    model_end_year: int = 2057
    discount_rate: float = 0.055
    exit_discount_rate: float = 0.055
    income_method_horizon_years: int = 30
    water_price_start: float = 75.0
    water_price_escalation: float = 0.025
    opex_monthly: float = 230_000.0
    opex_escalation: float = 0.025
    renewal_interval_years: int = 10
    renewal_capex_pct: float = 0.10
    renewal_included_in_opex: bool = True
    terminal_growth_rate: float = 0.0
    total_area_m2: float = 208_200.0
    total_consumption_m3_day: float = 160.0
    potable_water_share: float = 1.0
    operating_days: int = 365
    residual_life_years: int = 30
    ebitda_multiple_low: float = 8.0
    ebitda_multiple_base: float = 11.0
    ebitda_multiple_high: float = 14.0

    @property
    def full_potable_m3_day(self) -> float:
        return self.total_consumption_m3_day * self.potable_water_share


def _clean_halls(halls: pd.DataFrame) -> pd.DataFrame:
    df = halls.copy()
    required = ["Hala", "Plocha_m2", "Rok_spusteni"]
    for col in required:
        if col not in df.columns:
            df[col] = "" if col == "Hala" else 0
    df = df[["Hala", "Plocha_m2", "Rok_spusteni"]].copy()
    df["Hala"] = df["Hala"].fillna("").astype(str)
    df["Plocha_m2"] = pd.to_numeric(df["Plocha_m2"], errors="coerce").fillna(0.0)
    df["Rok_spusteni"] = pd.to_numeric(df["Rok_spusteni"], errors="coerce").fillna(0).astype(int)
    df = df[df["Plocha_m2"] > 0].reset_index(drop=True)
    hall_number = pd.to_numeric(df["Hala"].str.extract(r"(\d+)")[0], errors="coerce").fillna(9999)
    df = df.assign(_hall_number=hall_number)
    return df.sort_values(["_hall_number", "Hala"], ascending=[True, True]).drop(columns=["_hall_number"]).reset_index(drop=True)

def _safe_irr(cash_flows: Iterable[float]) -> Optional[float]:
    values = list(cash_flows)
    if not values or all(v >= 0 for v in values) or all(v <= 0 for v in values):
        return None
    try:
        if npf is not None:
            out = npf.irr(values)
            if np.isfinite(out):
                return float(out)
    except Exception:
        pass

    # Simple bisection fallback for yearly IRR.
    def npv(rate: float) -> float:
        return sum(v / ((1 + rate) ** i) for i, v in enumerate(values))

    low, high = -0.95, 1.0
    try:
        f_low, f_high = npv(low), npv(high)
        if f_low * f_high > 0:
            return None
        for _ in range(100):
            mid = (low + high) / 2
            f_mid = npv(mid)
            if abs(f_mid) < 1e-6:
                return mid
            if f_low * f_mid <= 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return (low + high) / 2
    except Exception:
        return None


def _first_year_where(df: pd.DataFrame, col: str, threshold: float = 0.0) -> Optional[int]:
    sub = df[df[col] >= threshold]
    if sub.empty:
        return None
    return int(sub.iloc[0]["Rok"])


def _years_from_start(year: Optional[int], start_year: int) -> Optional[int]:
    """Return the number of years from the investment year to a given calendar year."""
    if year is None:
        return None
    return max(0, int(year) - int(start_year))


def _years_label(years: Optional[int]) -> str:
    if years is None:
        return "není v modelu"
    if years == 1:
        return "1 rok"
    if 2 <= years <= 4:
        return f"{years} roky"
    return f"{years} let"


def _money(x: Optional[float], currency: str = "Kč") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:,.0f} {currency}".replace(",", " ")


def _pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.1%}"


def calculate_model(
    inputs: ModelInputs,
    halls: pd.DataFrame = DEFAULT_HALLS,
    opex: pd.DataFrame = DEFAULT_OPEX,
    sensitivity_prices: Optional[List[float]] = None,
    sensitivity_potable_shares: Optional[List[float]] = None,
) -> Dict[str, object]:
    halls = _clean_halls(halls)
    if sensitivity_prices is None:
        sensitivity_prices = [70, 75, 80, 85, 90]
    if sensitivity_potable_shares is None:
        sensitivity_potable_shares = [0.55, 0.65, 0.75]

    total_area = inputs.total_area_m2 if inputs.total_area_m2 > 0 else float(halls["Plocha_m2"].sum())
    total_area = max(total_area, 1.0)
    full_potable_m3_day = inputs.total_consumption_m3_day * inputs.potable_water_share

    years = list(range(inputs.capex_year, inputs.model_end_year + 1))
    rows = []
    for year in years:
        idx = year - inputs.capex_year
        if year < inputs.revenue_start_year:
            area_online = 0.0
        else:
            area_online = min(total_area, float(halls.loc[halls["Rok_spusteni"] <= year, "Plocha_m2"].sum()))
        pct_area = area_online / total_area
        potable_m3_day = pct_area * full_potable_m3_day
        potable_m3_year = potable_m3_day * inputs.operating_days
        if year < inputs.revenue_start_year:
            price = 0.0
            revenue = 0.0
            renewal_reserve_year = 0.0
            opex_year = 0.0
        else:
            exp_year = year - inputs.revenue_start_year
            price = inputs.water_price_start * ((1 + inputs.water_price_escalation) ** exp_year)
            revenue = potable_m3_year * price
            annual_renewal_base = 0.0 if getattr(inputs, "renewal_included_in_opex", False) else (
                inputs.capex_amount * inputs.renewal_capex_pct / int(inputs.renewal_interval_years)
                if int(inputs.renewal_interval_years) > 0 and float(inputs.renewal_capex_pct) > 0
                else 0.0
            )
            renewal_reserve_year = annual_renewal_base * ((1 + inputs.opex_escalation) ** exp_year)
            # Obnova technologie se nyní neúčtuje jednorázově po intervalu, ale jako
            # rovnoměrná roční rezerva zahrnutá v OPEX.
            opex_year = (inputs.opex_monthly * 12 + annual_renewal_base) * ((1 + inputs.opex_escalation) ** exp_year)
        capex = -inputs.capex_amount if year == inputs.capex_year else 0.0
        fcf = revenue - opex_year + capex
        dfactor = 1 / ((1 + inputs.discount_rate) ** idx)
        rows.append(
            {
                "Rok": year,
                "Index": idx,
                "Plocha v provozu (m²)": area_online,
                "% plochy": pct_area,
                "Pitná voda (m³/den)": potable_m3_day,
                "Pitná voda (m³/rok)": potable_m3_year,
                "Cena vody (Kč/m³)": price,
                "Výnosy (Kč)": revenue,
                "OPEX (Kč)": opex_year,
                "Obnova technologie v OPEX (Kč)": renewal_reserve_year,
                "CAPEX (Kč)": capex,
                "FCF (Kč)": fcf,
                "Měsíční FCF (Kč/měs.)": fcf / 12,
                "DF": dfactor,
                "PV FCF (Kč)": fcf * dfactor,
                "EBITDA (Kč)": revenue - opex_year,
            }
        )
    dcf = pd.DataFrame(rows)
    dcf["Kumul. FCF"] = dcf["FCF (Kč)"].cumsum()
    dcf["Kumul. PV"] = dcf["PV FCF (Kč)"].cumsum()

    npv_holding = float(dcf["PV FCF (Kč)"].sum())
    irr = _safe_irr(dcf["FCF (Kč)"].tolist())
    payback = _first_year_where(dcf, "Kumul. FCF", 0.0)
    discounted_payback = _first_year_where(dcf, "Kumul. PV", 0.0)
    payback_years = _years_from_start(payback, inputs.capex_year)
    discounted_payback_years = _years_from_start(discounted_payback, inputs.capex_year)
    net_cash_yield = float(dcf["FCF (Kč)"].sum())
    investment_life_years = max(0, int(inputs.model_end_year) - int(inputs.revenue_start_year) + 1)
    full_area_rows = dcf[dcf["Plocha v provozu (m²)"] >= total_area - 1e-6]
    first_full_year = int(full_area_rows.iloc[0]["Rok"]) if not full_area_rows.empty else None
    fcf_first_full_year = None
    monthly_fcf_first_full_year = None
    if first_full_year is not None:
        r = dcf.loc[dcf["Rok"] == first_full_year].iloc[0]
        fcf_first_full_year = float(r["FCF (Kč)"])
        monthly_fcf_first_full_year = float(r["Měsíční FCF (Kč/měs.)"])

    # Hall detail with derived full-year revenue at starting price.
    halls_out = halls.copy()
    halls_out["Podíl plochy"] = halls_out["Plocha_m2"] / total_area
    halls_out["Pitná voda (m³/den)"] = halls_out["Podíl plochy"] * full_potable_m3_day
    halls_out["Pitná voda (m³/rok)"] = halls_out["Pitná voda (m³/den)"] * inputs.operating_days
    halls_out["Roční výnos při ceně start"] = halls_out["Pitná voda (m³/rok)"] * inputs.water_price_start

    # Ramp by year for operational years.
    ramp_rows = []
    for year in range(inputs.revenue_start_year, min(inputs.revenue_start_year + 5, inputs.model_end_year) + 1):
        area_online = min(total_area, float(halls.loc[halls["Rok_spusteni"] <= year, "Plocha_m2"].sum()))
        pct_area = area_online / total_area
        potable_m3_day = pct_area * full_potable_m3_day
        ramp_rows.append(
            {
                "Rok": year,
                "Plocha v provozu (m²)": area_online,
                "% plochy": pct_area,
                "Pitná voda (m³/den)": potable_m3_day,
                "Pitná voda (m³/rok)": potable_m3_day * inputs.operating_days,
                "Výnos při startovní ceně": potable_m3_day * inputs.operating_days * inputs.water_price_start,
            }
        )
    ramp = pd.DataFrame(ramp_rows)

    # Income approach for exit value.
    # Assumption: the seller receives FCF in the sale year and sells at the end of that year.
    # The buyer pays for future FCF after the sale year, discounted back to the sale date.
    # The income-method value is intentionally bounded by the explicit investment life in
    # the DCF model. It therefore includes only future cash flows through model_end_year
    # and does not add a perpetuity/terminal value. This keeps the exit price conservative
    # for investor presentation even though the asset may technically generate cash flow
    # after the model horizon.
    exit_rate = max(float(inputs.exit_discount_rate), -0.95)
    income_horizon = max(1, int(inputs.income_method_horizon_years))

    def _project_operating_cash_flow(year: int) -> Tuple[float, float]:
        """Return projected FCF and EBITDA for any future year, including years beyond the explicit DCF model."""
        if year < inputs.revenue_start_year:
            return 0.0, 0.0
        area_online = min(total_area, float(halls.loc[halls["Rok_spusteni"] <= year, "Plocha_m2"].sum()))
        pct_area = area_online / total_area
        potable_m3_day = pct_area * full_potable_m3_day
        potable_m3_year = potable_m3_day * inputs.operating_days
        exp_year = year - inputs.revenue_start_year
        price = inputs.water_price_start * ((1 + inputs.water_price_escalation) ** exp_year)
        revenue = potable_m3_year * price
        annual_renewal_base = 0.0 if getattr(inputs, "renewal_included_in_opex", False) else (
            inputs.capex_amount * inputs.renewal_capex_pct / int(inputs.renewal_interval_years)
            if int(inputs.renewal_interval_years) > 0 and float(inputs.renewal_capex_pct) > 0
            else 0.0
        )
        opex_year = (inputs.opex_monthly * 12 + annual_renewal_base) * ((1 + inputs.opex_escalation) ** exp_year)
        ebitda = revenue - opex_year
        return float(ebitda), float(ebitda)

    def _income_value_at_sale_date(sale_year: int) -> float:
        value = 0.0
        last_valued_year = min(int(inputs.model_end_year), int(sale_year) + income_horizon)
        if last_valued_year <= sale_year:
            return 0.0
        for future_year in range(sale_year + 1, last_valued_year + 1):
            years_after_sale = int(future_year - sale_year)
            future_fcf, _ = _project_operating_cash_flow(future_year)
            value += future_fcf / ((1 + exit_rate) ** years_after_sale)
        return max(0.0, float(value))

    exit_rows = []
    op_years = range(1, inputs.model_end_year - inputs.revenue_start_year + 2)
    for op_year in op_years:
        calendar_year = inputs.revenue_start_year + op_year - 1
        r = dcf.loc[dcf["Rok"] == calendar_year]
        if r.empty:
            continue
        r = r.iloc[0]
        dfactor = float(r["DF"])
        cum_pv = float(r["Kumul. PV"])
        fcf = float(r["FCF (Kč)"])
        ebitda = float(r["EBITDA (Kč)"])
        residual = max(0.0, inputs.capex_amount * (inputs.residual_life_years - op_year) / max(inputs.residual_life_years, 1))
        yield_price = _income_value_at_sale_date(calendar_year)
        base_ebitda_price = max(0.0, ebitda * inputs.ebitda_multiple_base)
        low_ebitda_price = max(0.0, ebitda * inputs.ebitda_multiple_low)
        high_ebitda_price = max(0.0, ebitda * inputs.ebitda_multiple_high)
        exit_rows.append(
            {
                "Rok provozu": op_year,
                "Kalendářní rok": calendar_year,
                "FCF v roce (Kč)": fcf,
                "Kumul. PV CF do prodeje (Kč)": cum_pv,
                "Zůstatková hodnota (Kč)": residual,
                "Prodej za výnosovou metodu (Kč)": yield_price,
                "NPV při zůstatkové hodnotě (Kč)": cum_pv + residual * dfactor,
                "NPV při výnosové metodě (Kč)": cum_pv + yield_price * dfactor,
                "Min. prodejní cena pro NPV=0 (Kč)": max(0.0, -cum_pv / dfactor) if dfactor != 0 else 0.0,
                "EBITDA v roce (Kč)": ebitda,
                "EBITDA multiple": inputs.ebitda_multiple_base,
                f"Cena dle {inputs.ebitda_multiple_base:.1f}x EBITDA (Kč)": base_ebitda_price,
                f"Cena dle {inputs.ebitda_multiple_low:.1f}x EBITDA (Kč)": low_ebitda_price,
                f"Cena dle {inputs.ebitda_multiple_high:.1f}x EBITDA (Kč)": high_ebitda_price,
                f"NPV při {inputs.ebitda_multiple_base:.1f}x EBITDA (Kč)": cum_pv + base_ebitda_price * dfactor,
                "Poznámka": "EBITDA záporná – multiple metoda nedává smysl bez normalizace." if ebitda < 0 else "",
            }
        )
    exit_analysis = pd.DataFrame(exit_rows)

    # Sensitivity table: NPV holding for changes in water price and potable water share.
    def sensitivity_npv(price: float, share: float) -> float:
        altered = ModelInputs(**asdict(inputs))
        altered.water_price_start = float(price)
        altered.potable_water_share = float(share)
        altered.full_potable_m3_day  # keep dataclass property accessible
        d = calculate_dcf_only(altered, halls)["PV FCF (Kč)"].sum()
        return float(d)

    sens_rows = []
    for share in sensitivity_potable_shares:
        row = {"Pitná voda / cena vody": share}
        for price in sensitivity_prices:
            row[float(price)] = sensitivity_npv(price, share)
        sens_rows.append(row)
    sensitivity = pd.DataFrame(sens_rows)

    opex_out = opex.copy()
    for col in ["Oblast", "Mesicne_Kc", "Typ / poznámka"]:
        if col not in opex_out.columns:
            opex_out[col] = "" if col != "Mesicne_Kc" else 0.0
    opex_out = opex_out[["Oblast", "Mesicne_Kc", "Typ / poznámka"]].copy()
    opex_out["Oblast"] = opex_out["Oblast"].fillna("").astype(str)
    opex_out["Typ / poznámka"] = opex_out["Typ / poznámka"].fillna("").astype(str)
    opex_out["Mesicne_Kc"] = pd.to_numeric(opex_out["Mesicne_Kc"], errors="coerce").fillna(0.0)
    opex_out = opex_out[(opex_out["Oblast"].str.strip() != "") | (opex_out["Mesicne_Kc"] != 0) | (opex_out["Typ / poznámka"].str.strip() != "")].reset_index(drop=True)
    if opex_out.empty:
        opex_out = DEFAULT_OPEX.copy()
        opex_out["Mesicne_Kc"] = pd.to_numeric(opex_out["Mesicne_Kc"], errors="coerce").fillna(0.0)
    opex_out["Ročně 2028 (Kč)"] = opex_out["Mesicne_Kc"] * 12
    total_opex = max(float(opex_out["Mesicne_Kc"].sum()), 1.0)
    opex_out["% OPEX"] = opex_out["Mesicne_Kc"] / total_opex

    # Dashboard tables.
    exit_variants = []
    for op_year in [3, 5, 7, 10]:
        row = exit_analysis.loc[exit_analysis["Rok provozu"] == op_year]
        if row.empty:
            continue
        row = row.iloc[0]
        exit_variants.append(
            {
                "Exit varianta": f"Prodej po {op_year} letech provozu",
                "Rok prodeje": int(row["Kalendářní rok"]),
                "Min. cena NPV=0": float(row["Min. prodejní cena pro NPV=0 (Kč)"]),
                "Výnosová cena": float(row["Prodej za výnosovou metodu (Kč)"]),
                f"Cena dle {inputs.ebitda_multiple_base:.1f}x EBITDA": float(row[f"Cena dle {inputs.ebitda_multiple_base:.1f}x EBITDA (Kč)"]),
                "PV výnosové ceny": float(row["Prodej za výnosovou metodu (Kč)"] * dcf.loc[dcf["Rok"] == row["Kalendářní rok"], "DF"].iloc[0]),
                "NPV při prodeji": float(row["NPV při výnosové metodě (Kč)"]),
                "NPV držení": npv_holding,
            }
        )
    exit_variants = pd.DataFrame(exit_variants)

    base_multiple_label = f"{inputs.ebitda_multiple_base:g}".replace(".", ",")
    sale_cashflow_rows = []
    for sale_op_year in [5, 10]:
        sale_row = exit_analysis.loc[exit_analysis["Rok provozu"] == sale_op_year]
        if sale_row.empty:
            continue
        sale_row = sale_row.iloc[0]
        sale_year = int(sale_row["Kalendářní rok"])
        sale_dcf = dcf.loc[dcf["Rok"] == sale_year]
        if sale_dcf.empty:
            continue
        sale_dfactor = float(sale_dcf["DF"].iloc[0])
        income_sale_price = float(sale_row["Prodej za výnosovou metodu (Kč)"])
        ebitda_11_price = float(sale_row[f"Cena dle {inputs.ebitda_multiple_base:.1f}x EBITDA (Kč)"])
        cumulative_fcf = float(dcf.loc[dcf["Rok"] <= sale_year, "FCF (Kč)"].sum())
        cumulative_pv_fcf = float(dcf.loc[dcf["Rok"] <= sale_year, "PV FCF (Kč)"].sum())
        pv_income_sale = income_sale_price * sale_dfactor
        pv_ebitda_sale = ebitda_11_price * sale_dfactor
        sale_cashflow_rows.append(
            {
                "Scénář": f"Prodej v {sale_op_year}. roce provozu",
                "Rok provozu": sale_op_year,
                "Kal. rok prodeje": sale_year,
                "Kumul. Free CF do prodeje": cumulative_fcf,
                "PV Free CF do prodeje": cumulative_pv_fcf,
                "Prodejní cena – výnosová metoda": income_sale_price,
                "Celkové CF – výnosová metoda": cumulative_fcf + income_sale_price,
                "PV celkem – výnosová metoda": cumulative_pv_fcf + pv_income_sale,
                f"Prodejní cena – EBITDA {base_multiple_label}x": ebitda_11_price,
                f"Celkové CF – EBITDA {base_multiple_label}x": cumulative_fcf + ebitda_11_price,
                f"PV celkem – EBITDA {base_multiple_label}x": cumulative_pv_fcf + pv_ebitda_sale,
            }
        )
    sale_cashflow = pd.DataFrame(sale_cashflow_rows)

    # Two investor-facing tables for the CF + sale view. Keeping them separate avoids
    # a wide horizontal table and makes the 5-year / 10-year exit scenarios easier to read.
    if sale_cashflow.empty:
        sale_cashflow_income = pd.DataFrame()
        sale_cashflow_ebitda = pd.DataFrame()
    else:
        sale_cashflow_income = sale_cashflow[[
            "Scénář",
            "Rok provozu",
            "Kal. rok prodeje",
            "Kumul. Free CF do prodeje",
            "PV Free CF do prodeje",
            "Prodejní cena – výnosová metoda",
            "Celkové CF – výnosová metoda",
            "PV celkem – výnosová metoda",
        ]].rename(columns={
            "Kumul. Free CF do prodeje": "Kumul. FCF (Kč)",
            "PV Free CF do prodeje": "PV FCF (Kč)",
            "Prodejní cena – výnosová metoda": "Cena prodeje (Kč)",
            "Celkové CF – výnosová metoda": "CF + prodej (Kč)",
            "PV celkem – výnosová metoda": "PV celkem (Kč)",
        })
        sale_cashflow_ebitda = sale_cashflow[[
            "Scénář",
            "Rok provozu",
            "Kal. rok prodeje",
            "Kumul. Free CF do prodeje",
            "PV Free CF do prodeje",
            f"Prodejní cena – EBITDA {base_multiple_label}x",
            f"Celkové CF – EBITDA {base_multiple_label}x",
            f"PV celkem – EBITDA {base_multiple_label}x",
        ]].rename(columns={
            "Kumul. Free CF do prodeje": "Kumul. FCF (Kč)",
            "PV Free CF do prodeje": "PV FCF (Kč)",
            f"Prodejní cena – EBITDA {base_multiple_label}x": "Cena prodeje (Kč)",
            f"Celkové CF – EBITDA {base_multiple_label}x": "CF + prodej (Kč)",
            f"PV celkem – EBITDA {base_multiple_label}x": "PV celkem (Kč)",
        })

    metrics = {
        "NPV - čistá současná hodnota": npv_holding,
        "NPV držení do konce modelu": npv_holding,  # backward compatibility for older export fields
        "IRR": irr,
        "WACC": inputs.discount_rate,
        "Čistý výnos": net_cash_yield,
        "Doba životnosti investice": investment_life_years,
        "Bod zvratu": discounted_payback,
        "Bod zvratu - počet let": discounted_payback_years,
        "Diskontovaný bod zvratu": discounted_payback,
        "Diskontovaný bod zvratu - počet let": discounted_payback_years,
        "Nediskontovaná návratnost": payback,
        "Nediskontovaná návratnost - počet let": payback_years,
        "Diskontovaná návratnost": discounted_payback,
        "Diskontovaná návratnost - počet let": discounted_payback_years,
        "První plný rok provozu": first_full_year,
        "FCF v prvním plném roce": fcf_first_full_year,
        "Měsíční FCF v prvním plném roce": monthly_fcf_first_full_year,
        "Plná pitná kapacita v modelu (m³/den)": full_potable_m3_day,
        "Celková plocha hal (m²)": total_area,
        "Obnova technologie - interval": inputs.renewal_interval_years,
        "Obnova technologie - % CAPEX": inputs.renewal_capex_pct,
        "Oceňovací horizont výnosové metody": inputs.income_method_horizon_years,
    }

    return {
        "inputs": inputs,
        "dcf": dcf,
        "exit_analysis": exit_analysis,
        "exit_variants": exit_variants,
        "sale_cashflow": sale_cashflow,
        "sale_cashflow_income": sale_cashflow_income,
        "sale_cashflow_ebitda": sale_cashflow_ebitda,
        "sensitivity": sensitivity,
        "halls": halls_out,
        "ramp": ramp,
        "opex": opex_out,
        "sources": DEFAULT_SOURCES.copy(),
        "metrics": metrics,
    }


def calculate_dcf_only(inputs: ModelInputs, halls: pd.DataFrame = DEFAULT_HALLS) -> pd.DataFrame:
    halls = _clean_halls(halls)
    total_area = inputs.total_area_m2 if inputs.total_area_m2 > 0 else float(halls["Plocha_m2"].sum())
    total_area = max(total_area, 1.0)
    full_potable_m3_day = inputs.total_consumption_m3_day * inputs.potable_water_share
    rows = []
    for year in range(inputs.capex_year, inputs.model_end_year + 1):
        idx = year - inputs.capex_year
        area_online = 0.0 if year < inputs.revenue_start_year else min(total_area, float(halls.loc[halls["Rok_spusteni"] <= year, "Plocha_m2"].sum()))
        pct_area = area_online / total_area
        if year < inputs.revenue_start_year:
            revenue = 0.0
            opex_year = 0.0
            price = 0.0
        else:
            exp_year = year - inputs.revenue_start_year
            potable_m3_day = pct_area * full_potable_m3_day
            price = inputs.water_price_start * ((1 + inputs.water_price_escalation) ** exp_year)
            revenue = potable_m3_day * inputs.operating_days * price
            annual_renewal_base = 0.0 if getattr(inputs, "renewal_included_in_opex", False) else (
                inputs.capex_amount * inputs.renewal_capex_pct / int(inputs.renewal_interval_years)
                if int(inputs.renewal_interval_years) > 0 and float(inputs.renewal_capex_pct) > 0
                else 0.0
            )
            opex_year = (inputs.opex_monthly * 12 + annual_renewal_base) * ((1 + inputs.opex_escalation) ** exp_year)
        capex = -inputs.capex_amount if year == inputs.capex_year else 0.0
        fcf = revenue - opex_year + capex
        dfactor = 1 / ((1 + inputs.discount_rate) ** idx)
        rows.append({"Rok": year, "FCF (Kč)": fcf, "PV FCF (Kč)": fcf * dfactor})
    out = pd.DataFrame(rows)
    out["Kumul. PV"] = out["PV FCF (Kč)"].cumsum()
    return out


def create_excel_report(model: Dict[str, object]) -> bytes:
    """Create an investor-ready Excel export with formulas in the core model sheets.

    The Streamlit app calculates the same outputs for on-screen display. The Excel export below
    intentionally recreates the model logic with native Excel formulas so that users can audit
    and adjust assumptions directly in the downloaded workbook.
    """
    inputs: ModelInputs = model["inputs"]  # type: ignore[assignment]
    dcf_model: pd.DataFrame = model["dcf"].copy()  # type: ignore[assignment]
    exit_model: pd.DataFrame = model["exit_analysis"].copy()  # type: ignore[assignment]
    halls_model: pd.DataFrame = model["halls"].copy()  # type: ignore[assignment]
    opex_model: pd.DataFrame = model["opex"].copy()  # type: ignore[assignment]
    metrics: Dict[str, object] = model["metrics"]  # type: ignore[assignment]

    from io import BytesIO
    import xlsxwriter
    from xlsxwriter.utility import xl_rowcol_to_cell, xl_range

    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True})
    workbook.set_properties({
        "title": "Finanční analýza pro vodohospodářskou investici",
        "subject": "Investor-ready finanční model",
        "author": "ChatGPT",
        "comments": "Export z aplikace; hlavní výpočty jsou zapsané jako vzorce.",
    })
    try:
        workbook.set_calc_mode("auto")
    except Exception:
        pass

    # ----- Formats -----
    fmt_title = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#0B1320", "font_size": 13, "align": "left", "valign": "vcenter"})
    fmt_section = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E79", "align": "left", "valign": "vcenter"})
    fmt_header = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E79", "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True})
    fmt_input = workbook.add_format({"font_color": "#0000FF", "num_format": '#,##0;[Red](#,##0);-', "border": 1})
    fmt_input_pct = workbook.add_format({"font_color": "#0000FF", "num_format": '0.0%;[Red](0.0%);-', "border": 1})
    fmt_input_mult = workbook.add_format({"font_color": "#0000FF", "num_format": '0.0x', "border": 1})
    fmt_formula = workbook.add_format({"font_color": "#000000", "num_format": '#,##0;[Red](#,##0);-', "border": 1})
    fmt_formula_money = workbook.add_format({"font_color": "#000000", "num_format": '#,##0 "Kč";[Red](#,##0) "Kč";-', "border": 1})
    fmt_money = workbook.add_format({"num_format": '#,##0 "Kč";[Red](#,##0) "Kč";-', "border": 1})
    fmt_num = workbook.add_format({"num_format": '#,##0;[Red](#,##0);-', "border": 1})
    fmt_pct = workbook.add_format({"num_format": '0.0%;[Red](0.0%);-', "border": 1})
    fmt_mult = workbook.add_format({"num_format": '0.0x', "border": 1})
    fmt_text = workbook.add_format({"border": 1, "valign": "top", "text_wrap": True})
    fmt_formula_pct = workbook.add_format({"font_color": "#000000", "num_format": '0.0%;[Red](0.0%);-', "border": 1})
    fmt_note = workbook.add_format({"italic": True, "font_color": "#444444", "text_wrap": True, "valign": "top"})
    fmt_note_box = workbook.add_format({"italic": True, "font_color": "#444444", "text_wrap": True, "valign": "top", "bg_color": "#F2F2F2", "border": 1})
    fmt_kpi_label = workbook.add_format({"bold": True, "font_color": "#44546A", "align": "center", "valign": "vcenter", "bg_color": "#EAF2F8", "border": 1})
    fmt_kpi_value = workbook.add_format({"bold": True, "font_size": 13, "align": "center", "valign": "vcenter", "border": 1, "num_format": '#,##0;[Red](#,##0);-'})
    fmt_kpi_money = workbook.add_format({"bold": True, "font_size": 13, "align": "center", "valign": "vcenter", "border": 1, "num_format": '#,##0 "Kč";[Red](#,##0) "Kč";-'})
    fmt_kpi_pct = workbook.add_format({"bold": True, "font_size": 13, "align": "center", "valign": "vcenter", "border": 1, "num_format": '0.0%;[Red](0.0%);-'})

    def q(sheet_name: str) -> str:
        return "'" + sheet_name.replace("'", "''") + "'"

    def cell(row: int, col: int, abs_row: bool = False, abs_col: bool = False) -> str:
        return xl_rowcol_to_cell(row, col, row_abs=abs_row, col_abs=abs_col)

    def sheet_cell(sheet_name: str, row: int, col: int, abs_row: bool = True, abs_col: bool = True) -> str:
        return f"{q(sheet_name)}!{cell(row, col, abs_row=abs_row, abs_col=abs_col)}"

    def write_header(ws, row: int, headers: list[str]) -> None:
        for c, h in enumerate(headers):
            ws.write(row, c, h, fmt_header)
        ws.set_row(row, 30)

    def write_title(ws, title: str, last_col: int) -> None:
        ws.merge_range(0, 0, 0, last_col, title, fmt_title)
        ws.set_row(0, 26)

    def add_autofilter(ws, header_row: int, last_data_row: int, last_col: int) -> None:
        if last_data_row >= header_row:
            ws.autofilter(header_row, 0, last_data_row, last_col)

    def as_float(value, default=0.0) -> float:
        try:
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return default
            return float(value)
        except Exception:
            return default

    # Create Dashboard first so it appears as the first tab in Excel.
    ws_dash = workbook.add_worksheet("Dashboard")

    # ----- Assumptions -----
    ws_ass = workbook.add_worksheet("Assumptions")
    ws_ass.hide_gridlines(2)
    write_title(ws_ass, f"{inputs.project_name} – vstupy a předpoklady", 5)
    ws_ass.merge_range(1, 0, 1, 5, "Modře jsou tvrdě zadané vstupy. Černě jsou vzorce dopočítané v Excelu. OPEX měsíčně je navázán na součet položek v listu OPEX_detail.", fmt_note_box)
    ws_ass.set_row(1, 34)
    ws_ass.set_column("A:A", 16)
    ws_ass.set_column("B:B", 38)
    ws_ass.set_column("C:C", 18)
    ws_ass.set_column("D:D", 12)
    ws_ass.set_column("E:E", 44)
    ws_ass.set_column("F:F", 18)
    ass_headers = ["Kategorie", "Předpoklad / vstup", "Hodnota", "Typ", "Poznámka", "Excel vazba"]
    write_header(ws_ass, 2, ass_headers)

    ass_rows = [
        ["Model", "Měna", inputs.currency, "Input", "bez DPH, bez financování a před daní", "inp_currency", "text"],
        ["CAPEX", "Investice do systému", inputs.capex_amount, "Input", "uživatelský vstup", "inp_capex", "money"],
        ["CAPEX", "Rok investice", inputs.capex_year, "Input", "uživatelský vstup", "inp_capex_year", "year"],
        ["Provoz", "Začátek výnosů", inputs.revenue_start_year, "Input", "uživatelský vstup", "inp_start_year", "year"],
        ["Provoz", "Konec modelu", inputs.model_end_year, "Input", "uživatelský vstup", "inp_end_year", "year"],
        ["Provoz", "Doba životnosti investice", "=inp_end_year-inp_start_year+1", "Formula", "počet let provozu v modelu", "inp_investment_life", "formula_num", as_float(metrics.get("Doba životnosti investice"))],
        ["Diskont", "WACC / diskontní sazba", inputs.discount_rate, "Input", "uživatelský vstup", "inp_discount_rate", "pct"],
        ["Exit", "Diskontní sazba výnosové metody", inputs.exit_discount_rate, "Input", "požadovaný výnos kupujícího", "inp_exit_discount_rate", "pct"],
        ["Exit", "Oceňovací horizont výnosové metody", inputs.income_method_horizon_years, "Input", "horní limit let budoucích FCF; výpočet nepřekročí konec modelové životnosti", "inp_income_horizon_years", "year"],
        ["Cena", "Cena pitné vody", inputs.water_price_start, "Input", "Kč/m³ bez DPH", "inp_water_price_start", "num"],
        ["Cena", "Roční inflace ceny vody", inputs.water_price_escalation, "Input", "uživatelský vstup", "inp_water_price_inflation", "pct"],
        ["OPEX", "Fixní provozní náklady", "=SUM('OPEX_detail'!B:B)", "Formula", "součet položek v listu OPEX_detail", "inp_opex_monthly", "formula_money", inputs.opex_monthly],
        ["OPEX", "Roční inflace OPEX", inputs.opex_escalation, "Input", "uživatelský vstup", "inp_opex_inflation", "pct"],
        ["Obnova", "Interval obnovy technologie", inputs.renewal_interval_years, "Input", "let", "inp_renewal_interval", "year"],
        ["Obnova", "Obnova technologie", inputs.renewal_capex_pct, "Input", "% z CAPEX", "inp_renewal_pct", "pct"],
        ["Areál", "Celková plocha hal", inputs.total_area_m2, "Input", "m²", "inp_total_area", "num"],
        ["Voda", "Celková spotřeba při plném obsazení", inputs.total_consumption_m3_day, "Input", "m³/den", "inp_total_consumption", "num"],
        ["Voda", "Podíl pitné vody", inputs.potable_water_share, "Input", "%", "inp_potable_share", "pct"],
        ["Voda", "Pitná voda při plném obsazení", "=inp_total_consumption*inp_potable_share", "Formula", "m³/den", "inp_full_potable_m3_day", "formula_num", inputs.full_potable_m3_day],
        ["Voda", "Provozní dny", inputs.operating_days, "Input", "dny/rok", "inp_operating_days", "year"],
        ["Exit", "Životnost pro zůstatkovou hodnotu", inputs.residual_life_years, "Input", "let", "inp_residual_life", "year"],
        ["Exit", "EBITDA multiple – low", inputs.ebitda_multiple_low, "Input", "x", "inp_ebitda_low", "mult"],
        ["Exit", "EBITDA multiple – base", inputs.ebitda_multiple_base, "Input", "x", "inp_ebitda_base", "mult"],
        ["Exit", "EBITDA multiple – high", inputs.ebitda_multiple_high, "Input", "x", "inp_ebitda_high", "mult"],
    ]
    named_cells: dict[str, str] = {}
    for i, row in enumerate(ass_rows):
        r = 3 + i
        category, label, value, typ, note, name, kind = row[:7]
        cached = row[7] if len(row) > 7 else None
        ws_ass.write(r, 0, category, fmt_text)
        ws_ass.write(r, 1, label, fmt_text)
        if isinstance(value, str) and value.startswith("="):
            if kind == "formula_pct":
                fmt = fmt_formula_pct
            elif kind == "formula_money":
                fmt = fmt_formula_money
            else:
                fmt = fmt_formula
            ws_ass.write_formula(r, 2, value, fmt, cached if cached is not None else 0)
        else:
            if kind == "pct":
                fmt = fmt_input_pct
            elif kind == "mult":
                fmt = fmt_input_mult
            elif kind in ["money", "num", "year"]:
                fmt = fmt_input
            else:
                fmt = fmt_text
            ws_ass.write(r, 2, value, fmt)
        ws_ass.write(r, 3, typ, fmt_text)
        ws_ass.write(r, 4, note, fmt_text)
        ws_ass.write(r, 5, name, fmt_text)
        named_cells[name] = sheet_cell("Assumptions", r, 2)
        if typ == "Input":
            try:
                ws_ass.write_comment(r, 2, f"Zdroj: vstup z aplikace / uživatelský scénář. Poznámka: {note}")
            except Exception:
                pass
        ws_ass.set_row(r, 22)
    for name, ref in named_cells.items():
        workbook.define_name(name, f"={ref}")
    add_autofilter(ws_ass, 2, 2 + len(ass_rows), len(ass_headers) - 1)
    ws_ass.freeze_panes(3, 0)

    # ----- Halls and ramp -----
    ws_h = workbook.add_worksheet("Haly_nabeh")
    ws_h.hide_gridlines(2)
    write_title(ws_h, "Haly a náběh spotřeby pitné vody", 6)
    ws_h.merge_range(1, 0, 1, 6, "Náběh hal je vstupní část modelu. Modře jsou vstupy, černě dopočítané vzorce.", fmt_note_box)
    ws_h.set_row(1, 30)
    hall_headers = ["Hala", "Plocha (m²)", "Rok spuštění", "Podíl plochy", "Pitná voda (m³/den)", "Pitná voda (m³/rok)", "Roční výnos při startovní ceně (Kč)"]
    write_header(ws_h, 2, hall_headers)
    hall_start = 3
    halls_clean = _clean_halls(halls_model.rename(columns={"Plocha_m2": "Plocha_m2", "Rok_spusteni": "Rok_spusteni"}))
    if halls_clean.empty:
        halls_clean = _clean_halls(DEFAULT_HALLS)
    hall_end = hall_start + len(halls_clean) - 1
    hall_area_rng = f"{q('Haly_nabeh')}!{xl_range(hall_start, 1, hall_end, 1)}"
    hall_year_rng = f"{q('Haly_nabeh')}!{xl_range(hall_start, 2, hall_end, 2)}"
    for i, (_, row) in enumerate(halls_clean.iterrows()):
        r = hall_start + i
        ws_h.write(r, 0, row["Hala"], fmt_text)
        ws_h.write(r, 1, float(row["Plocha_m2"]), fmt_input)
        ws_h.write(r, 2, int(row["Rok_spusteni"]), fmt_input)
        d_formula = f"={cell(r,1)}/inp_total_area"
        e_formula = f"={cell(r,3)}*inp_full_potable_m3_day"
        f_formula = f"={cell(r,4)}*inp_operating_days"
        g_formula = f"={cell(r,5)}*inp_water_price_start"
        idx = i if i < len(model["halls"]) else None
        cached_vals = model["halls"].iloc[idx] if idx is not None else None
        ws_h.write_formula(r, 3, d_formula, fmt_formula_pct, float(cached_vals["Podíl plochy"]) if cached_vals is not None and "Podíl plochy" in cached_vals else 0)
        ws_h.write_formula(r, 4, e_formula, fmt_formula, float(cached_vals["Pitná voda (m³/den)"]) if cached_vals is not None and "Pitná voda (m³/den)" in cached_vals else 0)
        ws_h.write_formula(r, 5, f_formula, fmt_formula, float(cached_vals["Pitná voda (m³/rok)"]) if cached_vals is not None and "Pitná voda (m³/rok)" in cached_vals else 0)
        ws_h.write_formula(r, 6, g_formula, fmt_formula_money, float(cached_vals["Roční výnos při ceně start"]) if cached_vals is not None and "Roční výnos při ceně start" in cached_vals else 0)
    add_autofilter(ws_h, 2, hall_end, len(hall_headers) - 1)
    ws_h.set_column("A:A", 16)
    ws_h.set_column("B:G", 16)
    ws_h.freeze_panes(3, 0)

    ramp_header_row = hall_end + 4
    ws_h.merge_range(ramp_header_row - 2, 0, ramp_header_row - 2, 5, "Náběh podle roku", fmt_section)
    ramp_headers = ["Rok", "Plocha v provozu (m²)", "% plochy", "Pitná voda (m³/den)", "Pitná voda (m³/rok)", "Výnos při startovní ceně (Kč)"]
    write_header(ws_h, ramp_header_row, ramp_headers)
    ramp_start = ramp_header_row + 1
    ramp_years = list(range(int(inputs.revenue_start_year), min(int(inputs.revenue_start_year) + 5, int(inputs.model_end_year)) + 1))
    ramp_model = model.get("ramp", pd.DataFrame())
    for i, year in enumerate(ramp_years):
        r = ramp_start + i
        ws_h.write_formula(r, 0, f"=inp_start_year+{i}", fmt_formula, year)
        ws_h.write_formula(r, 1, f"=MIN(inp_total_area,SUMIFS({hall_area_rng},{hall_year_rng},\"<=\"&{cell(r,0)}))", fmt_formula, float(ramp_model.iloc[i]["Plocha v provozu (m²)"]) if isinstance(ramp_model, pd.DataFrame) and i < len(ramp_model) else 0)
        ws_h.write_formula(r, 2, f"={cell(r,1)}/inp_total_area", fmt_formula_pct, float(ramp_model.iloc[i]["% plochy"]) if isinstance(ramp_model, pd.DataFrame) and i < len(ramp_model) else 0)
        ws_h.write_formula(r, 3, f"={cell(r,2)}*inp_full_potable_m3_day", fmt_formula, float(ramp_model.iloc[i]["Pitná voda (m³/den)"]) if isinstance(ramp_model, pd.DataFrame) and i < len(ramp_model) else 0)
        ws_h.write_formula(r, 4, f"={cell(r,3)}*inp_operating_days", fmt_formula, float(ramp_model.iloc[i]["Pitná voda (m³/rok)"]) if isinstance(ramp_model, pd.DataFrame) and i < len(ramp_model) else 0)
        ws_h.write_formula(r, 5, f"={cell(r,4)}*inp_water_price_start", fmt_formula_money, float(ramp_model.iloc[i]["Výnos při startovní ceně"]) if isinstance(ramp_model, pd.DataFrame) and i < len(ramp_model) else 0)

    # ----- OPEX detail -----
    ws_o = workbook.add_worksheet("OPEX_detail")
    ws_o.hide_gridlines(2)
    write_title(ws_o, "Rozpad měsíčních a ročních provozních nákladů", 4)
    ws_o.merge_range(1, 0, 1, 4, "Měsíční částky jsou vstupy; roční částky a podíly jsou vzorce. Řádek Obnova technologie je běžná OPEX položka – částku lze upravit a změna se propíše do DCF přes součet OPEX.", fmt_note_box)
    ws_o.set_row(1, 42)
    opex_headers = ["Oblast", "Měsíčně (Kč)", "Typ / poznámka", "Ročně 2028 (Kč)", "% OPEX"]
    write_header(ws_o, 2, opex_headers)
    opex_start = 3
    opex_for_export = opex_model.copy()
    for col in ["Oblast", "Mesicne_Kc", "Typ / poznámka"]:
        if col not in opex_for_export.columns:
            opex_for_export[col] = "" if col != "Mesicne_Kc" else 0
    opex_for_export = opex_for_export[["Oblast", "Mesicne_Kc", "Typ / poznámka"]].copy()
    opex_for_export["Mesicne_Kc"] = pd.to_numeric(opex_for_export["Mesicne_Kc"], errors="coerce").fillna(0.0)
    opex_for_export = opex_for_export[opex_for_export["Mesicne_Kc"] != 0].reset_index(drop=True)
    if opex_for_export.empty:
        opex_for_export = DEFAULT_OPEX.copy()
    opex_end = opex_start + len(opex_for_export) - 1
    opex_monthly_range = f"{q('OPEX_detail')}!{xl_range(opex_start, 1, opex_end, 1)}"
    for i, (_, row) in enumerate(opex_for_export.iterrows()):
        r = opex_start + i
        oblast = str(row["Oblast"])
        ws_o.write(r, 0, oblast, fmt_text)
        ws_o.write(r, 1, float(row["Mesicne_Kc"]), fmt_input)
        ws_o.write(r, 2, row.get("Typ / poznámka", ""), fmt_text)
        ws_o.write_formula(r, 3, f"={cell(r,1)}*12", fmt_formula_money, float(row["Mesicne_Kc"]) * 12)
        ws_o.write_formula(r, 4, f"=IFERROR({cell(r,1)}/SUM({opex_monthly_range}),0)", fmt_formula_pct, float(row["Mesicne_Kc"]) / max(float(opex_for_export["Mesicne_Kc"].sum()), 1.0))
        ws_o.set_row(r, 24)
    add_autofilter(ws_o, 2, opex_end, len(opex_headers) - 1)
    ws_o.set_column("A:A", 34)
    ws_o.set_column("B:B", 17)
    ws_o.set_column("C:C", 64)
    ws_o.set_column("D:D", 18)
    ws_o.set_column("E:E", 12)
    ws_o.freeze_panes(3, 0)

    # ----- DCF -----
    ws_d = workbook.add_worksheet("DCF")
    ws_d.hide_gridlines(2)
    write_title(ws_d, "DCF – provozní cash flow", 16)
    ws_d.merge_range(1, 0, 1, 16, "Tento list obsahuje nativní Excel vzorce. OPEX je navázán na měsíční součet v listu OPEX_detail, který obsahuje i položku Obnova technologie.", fmt_note_box)
    ws_d.set_row(1, 34)
    dcf_headers = ["Rok", "Index", "Plocha v provozu (m²)", "% plochy", "Pitná voda (m³/den)", "Pitná voda (m³/rok)", "Cena vody (Kč/m³)", "Výnosy (Kč)", "OPEX vč. obnovy (Kč)", "CAPEX (Kč)", "FCF (Kč)", "Měsíční FCF (Kč/měs.)", "DF", "PV FCF (Kč)", "EBITDA (Kč)", "Kumul. FCF", "Kumul. PV"]
    write_header(ws_d, 2, dcf_headers)
    dcf_start = 3
    dcf_years = list(range(int(inputs.capex_year), int(inputs.model_end_year) + 1))
    dcf_end = dcf_start + len(dcf_years) - 1
    dcf_year_rng = f"{q('DCF')}!{xl_range(dcf_start, 0, dcf_end, 0)}"
    dcf_fcf_rng = f"{q('DCF')}!{xl_range(dcf_start, 10, dcf_end, 10)}"
    dcf_pv_rng = f"{q('DCF')}!{xl_range(dcf_start, 13, dcf_end, 13)}"
    dcf_ebitda_rng = f"{q('DCF')}!{xl_range(dcf_start, 14, dcf_end, 14)}"
    dcf_df_rng = f"{q('DCF')}!{xl_range(dcf_start, 12, dcf_end, 12)}"
    dcf_cum_fcf_rng = f"{q('DCF')}!{xl_range(dcf_start, 15, dcf_end, 15)}"
    dcf_cum_pv_rng = f"{q('DCF')}!{xl_range(dcf_start, 16, dcf_end, 16)}"
    opex_annual_formula = "inp_opex_monthly*12" if getattr(inputs, "renewal_included_in_opex", False) else "(inp_opex_monthly*12+IF(inp_renewal_interval>0,inp_capex*inp_renewal_pct/inp_renewal_interval,0))"
    for i, year in enumerate(dcf_years):
        r = dcf_start + i
        cache = dcf_model.iloc[i] if i < len(dcf_model) else {}
        ws_d.write_formula(r, 0, f"=inp_capex_year+{i}", fmt_formula, year)
        ws_d.write_formula(r, 1, f"={cell(r,0)}-inp_capex_year", fmt_formula, i)
        ws_d.write_formula(r, 2, f"=IF({cell(r,0)}<inp_start_year,0,MIN(inp_total_area,SUMIFS({hall_area_rng},{hall_year_rng},\"<=\"&{cell(r,0)})))", fmt_formula, as_float(cache.get("Plocha v provozu (m²)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 3, f"={cell(r,2)}/inp_total_area", fmt_formula_pct, as_float(cache.get("% plochy") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 4, f"={cell(r,3)}*inp_full_potable_m3_day", fmt_formula, as_float(cache.get("Pitná voda (m³/den)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 5, f"={cell(r,4)}*inp_operating_days", fmt_formula, as_float(cache.get("Pitná voda (m³/rok)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 6, f"=IF({cell(r,0)}<inp_start_year,0,inp_water_price_start*(1+inp_water_price_inflation)^({cell(r,0)}-inp_start_year))", fmt_formula, as_float(cache.get("Cena vody (Kč/m³)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 7, f"={cell(r,5)}*{cell(r,6)}", fmt_formula_money, as_float(cache.get("Výnosy (Kč)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 8, f"=IF({cell(r,0)}<inp_start_year,0,{opex_annual_formula}*(1+inp_opex_inflation)^({cell(r,0)}-inp_start_year))", fmt_formula_money, as_float(cache.get("OPEX (Kč)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 9, f"=IF({cell(r,0)}=inp_capex_year,-inp_capex,0)", fmt_formula_money, as_float(cache.get("CAPEX (Kč)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 10, f"={cell(r,7)}-{cell(r,8)}+{cell(r,9)}", fmt_formula_money, as_float(cache.get("FCF (Kč)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 11, f"={cell(r,10)}/12", fmt_formula_money, as_float(cache.get("Měsíční FCF (Kč/měs.)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 12, f"=1/(1+inp_discount_rate)^{cell(r,1)}", fmt_formula, as_float(cache.get("DF") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 13, f"={cell(r,10)}*{cell(r,12)}", fmt_formula_money, as_float(cache.get("PV FCF (Kč)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 14, f"={cell(r,7)}-{cell(r,8)}", fmt_formula_money, as_float(cache.get("EBITDA (Kč)") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 15, f"=SUM(${cell(dcf_start,10)}:{cell(r,10)})", fmt_formula_money, as_float(cache.get("Kumul. FCF") if hasattr(cache, 'get') else 0))
        ws_d.write_formula(r, 16, f"=SUM(${cell(dcf_start,13)}:{cell(r,13)})", fmt_formula_money, as_float(cache.get("Kumul. PV") if hasattr(cache, 'get') else 0))
    add_autofilter(ws_d, 2, dcf_end, len(dcf_headers) - 1)
    ws_d.freeze_panes(3, 1)
    ws_d.set_column("A:B", 8)
    ws_d.set_column("C:C", 12)
    ws_d.set_column("D:D", 8)
    ws_d.set_column("E:F", 12)
    ws_d.set_column("G:G", 11)
    ws_d.set_column("H:Q", 13)

    # ----- Helper sheet for income method -----
    ws_help = workbook.add_worksheet("Vypocty")
    ws_help.hide_gridlines(2)
    ws_help.write(0, 0, "Pomocný list pro výnosovou metodu", fmt_title)
    ws_help.write(1, 0, "List obsahuje budoucí FCF po roce prodeje a jejich diskontovanou hodnotu k roku prodeje. Výpočet je omezen koncem modelové životnosti investice; cash flow po tomto horizontu není do výnosové ceny započteno. Je skrytý, ale vzorce jsou v sešitu ponechané pro audit.", fmt_note)
    ws_help.write(3, 0, "Rok provozu", fmt_header)
    ws_help.write(3, 1, "Kal. rok prodeje", fmt_header)
    income_horizon = max(1, int(inputs.income_method_horizon_years))
    for i in range(1, income_horizon + 1):
        ws_help.write(3, 1 + i, i, fmt_header)
    income_value_col = 2 + income_horizon
    ws_help.write(3, income_value_col, "Výnosová cena", fmt_header)
    exit_op_years = list(range(1, int(inputs.model_end_year) - int(inputs.revenue_start_year) + 2))
    future_first_col = 2
    future_last_col = 1 + income_horizon
    helper_start = 4
    helper_end = helper_start + len(exit_op_years) - 1
    income_value_range = f"{q('Vypocty')}!{xl_range(helper_start, income_value_col, helper_end, income_value_col)}"
    helper_op_year_range = f"{q('Vypocty')}!{xl_range(helper_start, 0, helper_end, 0)}"
    for i, op_year in enumerate(exit_op_years):
        r = helper_start + i
        ws_help.write(r, 0, op_year, fmt_num)
        ws_help.write_formula(r, 1, f"=inp_start_year+{cell(r,0)}-1", fmt_formula, int(inputs.revenue_start_year) + op_year - 1)
        for h in range(1, income_horizon + 1):
            c = future_first_col + h - 1
            future_year_expr = f"{cell(r,1, abs_col=True)}+{cell(3,c, abs_row=True)}"
            # Future FCF is bounded by inp_end_year so the value follows the model life.
            future_fcf_formula = (
                f"=IF({future_year_expr}>inp_end_year,0,IF({future_year_expr}<inp_start_year,0,"
                f"(MIN(inp_total_area,SUMIFS({hall_area_rng},{hall_year_rng},\"<=\"&{future_year_expr}))/inp_total_area"
                f"*inp_full_potable_m3_day*inp_operating_days*inp_water_price_start*(1+inp_water_price_inflation)^({future_year_expr}-inp_start_year)"
                f"-{opex_annual_formula}*(1+inp_opex_inflation)^({future_year_expr}-inp_start_year))))"
            )
            # Cache the app's projected value when the year is visible in the current DCF table; otherwise leave 0 cache.
            ws_help.write_formula(r, c, future_fcf_formula, fmt_formula_money, 0)
        future_range = xl_range(r, future_first_col, r, future_last_col)
        horizon_range = xl_range(3, future_first_col, 3, future_last_col)
        cached_income = as_float(exit_model.iloc[i]["Prodej za výnosovou metodu (Kč)"]) if i < len(exit_model) else 0
        ws_help.write_formula(r, income_value_col, f"=MAX(0,SUMPRODUCT({future_range},1/(1+inp_exit_discount_rate)^{horizon_range}))", fmt_formula_money, cached_income)
    ws_help.set_column(0, 1, 14)
    ws_help.set_column(2, income_value_col, 14)
    ws_help.hide()

    # ----- Exit analysis -----
    ws_e = workbook.add_worksheet("Exit_analysis")
    ws_e.hide_gridlines(2)
    base_mult_label = f"{inputs.ebitda_multiple_base:g}".replace(".", ",")
    low_mult_label = f"{inputs.ebitda_multiple_low:g}".replace(".", ",")
    high_mult_label = f"{inputs.ebitda_multiple_high:g}".replace(".", ",")
    write_title(ws_e, "Exitová analýza – prodej po jednotlivých letech", 15)
    ws_e.merge_range(1, 0, 1, 15, "Výpočty jsou vzorci navázané na DCF a vstupní předpoklady. Výnosová metoda oceňuje projekt podle budoucích Free CF, které může kupující získat po koupi. Po konci nastavené životnosti už model další cash flow neoceňuje, proto je v posledním roce výnosová cena 0.", fmt_note_box)
    ws_e.set_row(1, 34)
    exit_headers = ["Rok provozu", "Kalendářní rok", "FCF v roce (Kč)", "Kumul. PV CF do prodeje (Kč)", "Zůstatková hodnota (Kč)", "Prodej za výnosovou metodu (Kč)", "NPV při zůstatkové hodnotě (Kč)", "NPV při výnosové metodě (Kč)", "Min. prodejní cena pro NPV=0 (Kč)", "EBITDA v roce (Kč)", "EBITDA multiple", f"Cena dle {base_mult_label}x EBITDA (Kč)", f"Cena dle {low_mult_label}x EBITDA (Kč)", f"Cena dle {high_mult_label}x EBITDA (Kč)", f"NPV při {base_mult_label}x EBITDA (Kč)", "Poznámka"]
    exit_header_row = 3
    exit_start = exit_header_row + 1
    write_header(ws_e, exit_header_row, exit_headers)
    exit_end = exit_start + len(exit_op_years) - 1
    exit_op_year_rng = f"{q('Exit_analysis')}!{xl_range(exit_start, 0, exit_end, 0)}"
    for i, op_year in enumerate(exit_op_years):
        r = exit_start + i
        cache = exit_model.iloc[i] if i < len(exit_model) else {}
        sale_year_cell = cell(r, 1)
        df_lookup = f"INDEX({dcf_df_rng},MATCH({sale_year_cell},{dcf_year_rng},0))"
        ws_e.write_formula(r, 0, f"=ROW()-{exit_start}", fmt_formula, op_year)
        ws_e.write_formula(r, 1, f"=inp_start_year+{cell(r,0)}-1", fmt_formula, int(inputs.revenue_start_year) + op_year - 1)
        ws_e.write_formula(r, 2, f"=INDEX({dcf_fcf_rng},MATCH({sale_year_cell},{dcf_year_rng},0))", fmt_formula_money, as_float(cache.get("FCF v roce (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 3, f"=INDEX({dcf_cum_pv_rng},MATCH({sale_year_cell},{dcf_year_rng},0))", fmt_formula_money, as_float(cache.get("Kumul. PV CF do prodeje (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 4, f"=MAX(0,inp_capex*(inp_residual_life-{cell(r,0)})/MAX(inp_residual_life,1))", fmt_formula_money, as_float(cache.get("Zůstatková hodnota (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 5, f"=INDEX({income_value_range},MATCH({cell(r,0)},{helper_op_year_range},0))", fmt_formula_money, as_float(cache.get("Prodej za výnosovou metodu (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 6, f"={cell(r,3)}+{cell(r,4)}*{df_lookup}", fmt_formula_money, as_float(cache.get("NPV při zůstatkové hodnotě (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 7, f"={cell(r,3)}+{cell(r,5)}*{df_lookup}", fmt_formula_money, as_float(cache.get("NPV při výnosové metodě (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 8, f"=MAX(0,-{cell(r,3)}/{df_lookup})", fmt_formula_money, as_float(cache.get("Min. prodejní cena pro NPV=0 (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 9, f"=INDEX({dcf_ebitda_rng},MATCH({sale_year_cell},{dcf_year_rng},0))", fmt_formula_money, as_float(cache.get("EBITDA v roce (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 10, "=inp_ebitda_base", fmt_formula, inputs.ebitda_multiple_base)
        ws_e.write_formula(r, 11, f"=MAX(0,{cell(r,9)}*inp_ebitda_base)", fmt_formula_money, as_float(cache.get(f"Cena dle {inputs.ebitda_multiple_base:.1f}x EBITDA (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 12, f"=MAX(0,{cell(r,9)}*inp_ebitda_low)", fmt_formula_money, as_float(cache.get(f"Cena dle {inputs.ebitda_multiple_low:.1f}x EBITDA (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 13, f"=MAX(0,{cell(r,9)}*inp_ebitda_high)", fmt_formula_money, as_float(cache.get(f"Cena dle {inputs.ebitda_multiple_high:.1f}x EBITDA (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 14, f"={cell(r,3)}+{cell(r,11)}*{df_lookup}", fmt_formula_money, as_float(cache.get(f"NPV při {inputs.ebitda_multiple_base:.1f}x EBITDA (Kč)") if hasattr(cache, 'get') else 0))
        ws_e.write_formula(r, 15, f"=IF({cell(r,9)}<0,\"EBITDA záporná – multiple metoda nedává smysl bez normalizace.\",\"\")", fmt_text, cache.get("Poznámka") if hasattr(cache, 'get') else "")
    add_autofilter(ws_e, exit_header_row, exit_end, len(exit_headers) - 1)
    ws_e.freeze_panes(exit_start, 2)
    ws_e.set_column("A:B", 11)
    ws_e.set_column("C:P", 16)
    ws_e.set_column("P:P", 40)

    # ----- CF + prodej -----
    ws_cf = workbook.add_worksheet("CF_prodej")
    ws_cf.hide_gridlines(2)
    write_title(ws_cf, "Cash flow a prodej – scénář prodeje po 5 a 10 letech", 7)
    ws_cf.merge_range(1, 0, 1, 7, "Tabulky ukazují pouze prodej v 5. a 10. roce provozu, včetně kumulovaného CF do prodeje. Výnosová metoda počítá budoucí Free CF jen do konce nastavené životnosti investice.", fmt_note_box)
    ws_cf.set_row(1, 32)
    cf_headers = ["Scénář", "Rok provozu", "Kal. rok prodeje", "Kumul. FCF (Kč)", "PV FCF (Kč)", "Cena prodeje (Kč)", "CF + prodej (Kč)", "PV celkem (Kč)"]
    cf_sale_years = [5, 10]
    cf_income = model.get("sale_cashflow_income", pd.DataFrame())
    cf_ebitda = model.get("sale_cashflow_ebitda", pd.DataFrame())

    def write_cf_table(title: str, start_row: int, method_col: int, cache_df: pd.DataFrame) -> int:
        ws_cf.merge_range(start_row, 0, start_row, 7, title, fmt_section)
        header_row = start_row + 1
        write_header(ws_cf, header_row, cf_headers)
        for i, op_year in enumerate(cf_sale_years):
            r = header_row + 1 + i
            sale_year_formula = f"=inp_start_year+{cell(r,1)}-1"
            sale_year_ref = cell(r, 2)
            df_lookup = f"INDEX({dcf_df_rng},MATCH({sale_year_ref},{dcf_year_rng},0))"
            cache = cache_df.iloc[i] if isinstance(cache_df, pd.DataFrame) and i < len(cache_df) else {}
            ws_cf.write_formula(r, 0, f"=\"Prodej v \"&{cell(r,1)}&\". roce provozu\"", fmt_text, f"Prodej v {op_year}. roce provozu")
            ws_cf.write(r, 1, op_year, fmt_input)
            ws_cf.write_formula(r, 2, sale_year_formula, fmt_formula, int(inputs.revenue_start_year) + op_year - 1)
            ws_cf.write_formula(r, 3, f"=SUMIFS({dcf_fcf_rng},{dcf_year_rng},\"<=\"&{sale_year_ref})", fmt_formula_money, as_float(cache.get("Kumul. FCF (Kč)") if hasattr(cache, 'get') else 0))
            ws_cf.write_formula(r, 4, f"=SUMIFS({dcf_pv_rng},{dcf_year_rng},\"<=\"&{sale_year_ref})", fmt_formula_money, as_float(cache.get("PV FCF (Kč)") if hasattr(cache, 'get') else 0))
            ws_cf.write_formula(r, 5, f"=INDEX({q('Exit_analysis')}!{xl_range(exit_start, method_col, exit_end, method_col)},MATCH({cell(r,1)},{exit_op_year_rng},0))", fmt_formula_money, as_float(cache.get("Cena prodeje (Kč)") if hasattr(cache, 'get') else 0))
            ws_cf.write_formula(r, 6, f"={cell(r,3)}+{cell(r,5)}", fmt_formula_money, as_float(cache.get("CF + prodej (Kč)") if hasattr(cache, 'get') else 0))
            ws_cf.write_formula(r, 7, f"={cell(r,4)}+{cell(r,5)}*{df_lookup}", fmt_formula_money, as_float(cache.get("PV celkem (Kč)") if hasattr(cache, 'get') else 0))
            ws_cf.set_row(r, 18)
        return header_row + 1 + len(cf_sale_years) + 2

    next_row = write_cf_table("1) Prodej výnosovou metodou", 3, 5, cf_income if isinstance(cf_income, pd.DataFrame) else pd.DataFrame())
    write_cf_table(f"2) Prodej metodou EBITDA {base_mult_label}x", next_row, 11, cf_ebitda if isinstance(cf_ebitda, pd.DataFrame) else pd.DataFrame())
    ws_cf.set_column("A:A", 28)
    ws_cf.set_column("B:C", 11)
    ws_cf.set_column("D:H", 17)

    # ----- Dashboard -----
    ws_dash.hide_gridlines(2)
    write_title(ws_dash, "Finanční analýza pro vodohospodářskou investici", 8)
    ws_dash.merge_range(1, 0, 1, 8, f"Projekt: {inputs.project_name}. Hodnoty jsou vzorci navázané na DCF a vstupy v sešitu. OPEX je navázán na rozpis nákladů v listu OPEX_detail.", fmt_note_box)
    ws_dash.set_row(1, 34)
    ws_dash.set_column("A:A", 28)
    ws_dash.set_column("B:B", 19)
    ws_dash.set_column("C:C", 52)
    ws_dash.set_column("D:I", 17)

    # KPI tiles.
    payback_years_formula = f'=IFERROR(INDEX({q("DCF")}!{xl_range(dcf_start,1,dcf_end,1)},MATCH(1,INDEX(({dcf_cum_pv_rng}>=0)*1,0),0)),"není v modelu")'
    kpis = [
        ("Doba návratnosti", payback_years_formula, metrics.get("Diskontovaná návratnost - počet let"), fmt_kpi_value),
        ("IRR", f"=IRR({dcf_fcf_rng})", metrics.get("IRR"), fmt_kpi_pct),
        ("WACC", "=inp_discount_rate", metrics.get("WACC"), fmt_kpi_pct),
        ("Doba životnosti", "=inp_investment_life", metrics.get("Doba životnosti investice"), fmt_kpi_value),
    ]
    for i, (label, formula, cached, fmt_value) in enumerate(kpis):
        col = i * 2
        ws_dash.merge_range(3, col, 3, col + 1, label, fmt_kpi_label)
        ws_dash.merge_range(4, col, 4, col + 1, "", fmt_value)
        ws_dash.write_formula(4, col, formula, fmt_value, cached if cached is not None else 0)
        ws_dash.set_column(col, col + 1, 14)

    ws_dash.merge_range(7, 0, 7, 2, "Klíčové vstupy", fmt_section)
    renewal_monthly_cached = 0.0
    try:
        renewal_rows = opex_model.loc[opex_model["Oblast"].astype(str).str.strip().str.lower() == "obnova technologie"]
        if not renewal_rows.empty:
            renewal_monthly_cached = float(pd.to_numeric(renewal_rows["Mesicne_Kc"], errors="coerce").fillna(0).iloc[0])
    except Exception:
        renewal_monthly_cached = 0.0
    dash_inputs = [
        ["CAPEX", "=inp_capex", f"investice v roce {inputs.capex_year}", fmt_money, inputs.capex_amount],
        ["Cena pitné vody", "=inp_water_price_start", "base case, Kč/m³ bez DPH", fmt_num, inputs.water_price_start],
        ["WACC / diskontní sazba", "=inp_discount_rate", "DCF model", fmt_pct, inputs.discount_rate],
        ["Plná pitná kapacita", "=inp_full_potable_m3_day", f"{inputs.potable_water_share:.0%} z {inputs.total_consumption_m3_day:.0f} m³/den", fmt_num, inputs.full_potable_m3_day],
        ["Celková plocha hal", "=inp_total_area", "dle náběhové tabulky", fmt_num, inputs.total_area_m2],
        ["Obnova technologie", '=SUMIF(\'OPEX_detail\'!A:A,"Obnova technologie",\'OPEX_detail\'!B:B)', "měsíčně v OPEX", fmt_money, renewal_monthly_cached],
    ]
    for i, (label, formula, note, fmt, cached) in enumerate(dash_inputs):
        r = 8 + i
        ws_dash.write(r, 0, label, fmt_text)
        ws_dash.write_formula(r, 1, formula, fmt, cached)
        ws_dash.write(r, 2, note, fmt_text)
        ws_dash.set_row(r, 22)

    ws_dash.merge_range(16, 0, 16, 2, "Výsledky držení", fmt_section)
    first_full_year_formula = f'=IFERROR(INDEX({q("DCF")}!{xl_range(dcf_start,0,dcf_end,0)},MATCH(1,INDEX(({q("DCF")}!{xl_range(dcf_start,2,dcf_end,2)}>=inp_total_area)*1,0),0)),"není v modelu")'
    result_rows = [
        ["Doba návratnosti", payback_years_formula, "počet let, diskontovaně", fmt_formula, metrics.get("Diskontovaná návratnost - počet let")],
        ["IRR", f"=IRR({dcf_fcf_rng})", "vnitřní výnosové procento", fmt_formula_pct, metrics.get("IRR")],
        ["WACC", "=inp_discount_rate", "diskontní sazba modelu", fmt_formula_pct, metrics.get("WACC")],
        ["Doba životnosti investice", "=inp_investment_life", "počet let provozu v modelu", fmt_formula, metrics.get("Doba životnosti investice")],
        ["NPV - čistá současná hodnota", f"=SUM({dcf_pv_rng})", "DCF držení do konce modelu", fmt_formula_money, metrics.get("NPV - čistá současná hodnota")],
        ["Diskontovaný bod zvratu", f'=IFERROR(INDEX({q("DCF")}!{xl_range(dcf_start,0,dcf_end,0)},MATCH(1,INDEX(({dcf_cum_pv_rng}>=0)*1,0),0)),"není v modelu")', "kalendářní rok, diskontovaně", fmt_formula, metrics.get("Diskontovaný bod zvratu") or 0],
        ["První plný rok provozu", first_full_year_formula, "dosažení plné plochy", fmt_formula, metrics.get("První plný rok provozu") or 0],
        ["FCF v prvním plném roce", f"=IFERROR(INDEX({dcf_fcf_rng},MATCH({cell(23,1)},{dcf_year_rng},0)),0)", "roční FCF", fmt_formula_money, metrics.get("FCF v prvním plném roce") or 0],
    ]
    for i, (label, formula, note, fmt, cached) in enumerate(result_rows):
        r = 17 + i
        ws_dash.write(r, 0, label, fmt_text)
        ws_dash.write_formula(r, 1, formula, fmt, cached if cached is not None else 0)
        ws_dash.write(r, 2, note, fmt_text)
        ws_dash.set_row(r, 22)

    ws_dash.merge_range(27, 0, 30, 8, "NPV ukazuje, kolik peněz (v dnešní hodnotě) projekt celkově vydělá nad rámec vložených prostředků. Kladná hodnota znamená, že projekt generuje hodnotu. IRR je vnitřní výnosové procento – diskontní sazba, při které je NPV projektu rovna nule. WACC je požadované zhodnocení kapitálu použité pro diskontování.", fmt_note_box)
    for _r in range(27, 31):
        ws_dash.set_row(_r, 22)

    ws_dash.merge_range(33, 0, 33, 6, "Exit – vybrané roky prodeje", fmt_section)
    dash_exit_headers = ["Rok provozu", "Kal. rok", "Výnosová cena", f"Cena EBITDA {base_mult_label}x", "PV výnosové ceny", "NPV při prodeji", "NPV držení"]
    write_header(ws_dash, 34, dash_exit_headers)
    dash_exit_years = [3, 5, 7, 10]
    for i, op_year in enumerate(dash_exit_years):
        r = 35 + i
        cal_year_formula = f"=inp_start_year+{cell(r,0)}-1"
        df_lookup = f"INDEX({dcf_df_rng},MATCH({cell(r,1)},{dcf_year_rng},0))"
        row_cache = model.get("exit_variants", pd.DataFrame())
        cache = row_cache.iloc[i] if isinstance(row_cache, pd.DataFrame) and i < len(row_cache) else {}
        ws_dash.write(r, 0, op_year, fmt_input)
        ws_dash.write_formula(r, 1, cal_year_formula, fmt_formula, int(inputs.revenue_start_year) + op_year - 1)
        ws_dash.write_formula(r, 2, f"=INDEX({q('Exit_analysis')}!{xl_range(exit_start,5,exit_end,5)},MATCH({cell(r,0)},{exit_op_year_rng},0))", fmt_formula_money, as_float(cache.get("Výnosová cena") if hasattr(cache, 'get') else 0))
        ws_dash.write_formula(r, 3, f"=INDEX({q('Exit_analysis')}!{xl_range(exit_start,11,exit_end,11)},MATCH({cell(r,0)},{exit_op_year_rng},0))", fmt_formula_money, as_float(cache.get(f"Cena dle {inputs.ebitda_multiple_base:.1f}x EBITDA") if hasattr(cache, 'get') else 0))
        ws_dash.write_formula(r, 4, f"={cell(r,2)}*{df_lookup}", fmt_formula_money, as_float(cache.get("PV výnosové ceny") if hasattr(cache, 'get') else 0))
        ws_dash.write_formula(r, 5, f"=INDEX({q('Exit_analysis')}!{xl_range(exit_start,7,exit_end,7)},MATCH({cell(r,0)},{exit_op_year_rng},0))", fmt_formula_money, as_float(cache.get("NPV při prodeji") if hasattr(cache, 'get') else 0))
        ws_dash.write_formula(r, 6, f"=SUM({dcf_pv_rng})", fmt_formula_money, metrics.get("NPV - čistá současná hodnota") or 0)
    ws_dash.freeze_panes(7, 0)

    # ----- Risk analysis -----
    ws_r = workbook.add_worksheet("Analyza_rizik")
    ws_r.hide_gridlines(2)
    write_title(ws_r, "Analýza rizik – dopad hlavních rizik na ekonomiku projektu", 6)
    ws_r.merge_range(1, 0, 1, 6, "Základní scénář je vzorcem navázán na model. Ostatní rizikové scénáře jsou přepočteny aplikací při exportu a slouží jako investorský přehled.", fmt_note_box)
    ws_r.set_row(1, 34)
    risk_df = pd.DataFrame(model.get("risk_analysis", model.get("sensitivity", pd.DataFrame())))
    risk_headers = list(risk_df.columns) if not risk_df.empty else ["Scénář", "Co se mění", "NPV", "Změna NPV oproti základnímu scénáři", "IRR", "Doba návratnosti", "Čistý výnos"]
    write_header(ws_r, 3, risk_headers)
    for i, (_, row) in enumerate(risk_df.iterrows()):
        r = 4 + i
        for c, h in enumerate(risk_headers):
            val = row.get(h, "")
            if i == 0 and h == "NPV":
                ws_r.write_formula(r, c, f"=SUM({dcf_pv_rng})", fmt_formula_money, as_float(val))
            elif i == 0 and h == "IRR":
                ws_r.write_formula(r, c, f"=IRR({dcf_fcf_rng})", fmt_formula_pct, as_float(val))
            elif i == 0 and h == "Čistý výnos":
                ws_r.write_formula(r, c, f"=SUM({dcf_fcf_rng})", fmt_formula_money, as_float(val))
            elif i == 0 and h == "Změna NPV oproti základnímu scénáři":
                ws_r.write_formula(r, c, "=0", fmt_formula_money, 0)
            elif isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                if h in ["NPV", "Změna NPV oproti základnímu scénáři", "Čistý výnos"]:
                    ws_r.write_number(r, c, float(val), fmt_money)
                elif h == "IRR":
                    ws_r.write_number(r, c, float(val), fmt_pct)
                else:
                    ws_r.write_number(r, c, float(val), fmt_num)
            else:
                ws_r.write(r, c, "" if val is None else str(val), fmt_text)
    ws_r.set_column("A:B", 34)
    ws_r.set_column("C:G", 18)
    if not risk_df.empty:
        add_autofilter(ws_r, 3, 3 + len(risk_df), len(risk_headers) - 1)

    # ----- Sources -----
    ws_s = workbook.add_worksheet("Sources")
    ws_s.hide_gridlines(2)
    write_title(ws_s, "Zdroje a poznámky", 3)
    sources = model.get("sources", DEFAULT_SOURCES.copy())
    sources_df = pd.DataFrame(sources)
    source_headers = list(sources_df.columns)
    write_header(ws_s, 2, source_headers)
    for i, (_, row) in enumerate(sources_df.iterrows()):
        r = 3 + i
        for c, h in enumerate(source_headers):
            ws_s.write(r, c, row.get(h, ""), fmt_text)
    ws_s.set_column("A:D", 34)
    if not sources_df.empty:
        add_autofilter(ws_s, 2, 2 + len(sources_df), len(source_headers) - 1)

    # Sheet order is determined by creation order. Put Dashboard first by hiding/reordering is not supported directly,
    # so we created it after core sheets. Keep workbook usable via tabs; selected worksheet is Dashboard.
    try:
        ws_dash.activate()
        ws_dash.select()
    except Exception:
        pass

    # Final formatting tweaks.
    for ws in [ws_dash, ws_ass, ws_h, ws_o, ws_d, ws_e, ws_cf, ws_r, ws_s]:
        try:
            ws.set_landscape()
            ws.fit_to_pages(1, 0)
        except Exception:
            pass

    workbook.close()
    output.seek(0)
    return output.read()

def summarize_metrics(model: Dict[str, object]) -> pd.DataFrame:
    inputs: ModelInputs = model["inputs"]  # type: ignore[assignment]
    m = model["metrics"]  # type: ignore[assignment]
    break_even_year = m.get("Diskontovaný bod zvratu")
    break_even_years = m.get("Diskontovaný bod zvratu - počet let")
    if break_even_year:
        break_even_value = f"{int(break_even_year)} ({_years_label(break_even_years)})"
    else:
        break_even_value = "není v modelu"
    return pd.DataFrame(
        [
            {"Metrika": "Doba návratnosti", "Hodnota": _years_label(m.get("Diskontovaná návratnost - počet let"))},
            {"Metrika": "IRR", "Hodnota": _pct(m["IRR"])},
            {"Metrika": "WACC", "Hodnota": _pct(m["WACC"])},
            {"Metrika": "Doba životnosti investice", "Hodnota": _years_label(m.get("Doba životnosti investice"))},
            {"Metrika": "NPV - čistá současná hodnota", "Hodnota": _money(m["NPV - čistá současná hodnota"], inputs.currency)},
            {"Metrika": "Diskontovaný bod zvratu", "Hodnota": break_even_value},
            {"Metrika": "První plný rok provozu", "Hodnota": f"{m['První plný rok provozu']}" if m["První plný rok provozu"] else "není v modelu"},
            {"Metrika": "FCF v prvním plném roce", "Hodnota": _money(m["FCF v prvním plném roce"], inputs.currency)},
            {"Metrika": "Měsíční FCF v prvním plném roce", "Hodnota": _money(m["Měsíční FCF v prvním plném roce"], inputs.currency)},
        ]
    )
