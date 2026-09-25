"""
Step 8: an ROI calculator for a process model (Chapter 15).

The calculation structure is general - plug in your own numbers (cost per person-month,
batch value, deviation frequency), the model stays the same. Below: ILLUSTRATIVE
numbers from the book "Process Modeling in Pharma: From Zero to a Validated Model" (the bioreactor DO model), to verify that the calculator
reproduces their result (payback ~2.9 years).
"""
from dataclasses import dataclass


@dataclass
class DevelopmentCost:
    person_months: float
    rate_per_person_month: float
    one_time_infrastructure: float

    @property
    def total(self):
        return self.person_months * self.rate_per_person_month + self.one_time_infrastructure


@dataclass
class AnnualMaintenanceCost:
    monitoring_retraining: float
    periodic_requalification: float
    infrastructure: float

    @property
    def total(self):
        return self.monitoring_retraining + self.periodic_requalification + self.infrastructure


@dataclass
class AnnualBenefits:
    avoided_formal_deviations: float   # count/year x investigation/CAPA cost
    avoided_batch_loss: float          # batch value / expected time between losses

    @property
    def total(self):
        return self.avoided_formal_deviations + self.avoided_batch_loss


def calculate_roi(dev_cost: DevelopmentCost, maint_cost: AnnualMaintenanceCost,
                   benefits: AnnualBenefits, batches_per_year: int, npv_years: int = 5,
                   discount_rate: float = 0.08):
    annual_net_benefit = benefits.total - maint_cost.total
    payback_years = dev_cost.total / annual_net_benefit if annual_net_benefit > 0 else float("inf")

    # NPV: sum of discounted net benefits over `npv_years`, minus the initial cost
    npv = -dev_cost.total
    for year in range(1, npv_years + 1):
        npv += annual_net_benefit / ((1 + discount_rate) ** year)

    cost_per_batch = maint_cost.total / batches_per_year
    benefit_per_batch = benefits.total / batches_per_year
    net_per_batch = benefit_per_batch - cost_per_batch

    return {
        "development_cost": dev_cost.total,
        "annual_maintenance_cost": maint_cost.total,
        "total_annual_benefit": benefits.total,
        "annual_net_benefit": annual_net_benefit,
        "payback_years": payback_years,
        "payback_months": payback_years * 12,
        f"npv_{npv_years}yr": npv,
        "cost_per_batch": cost_per_batch,
        "benefit_per_batch": benefit_per_batch,
        "net_per_batch": net_per_batch,
    }


if __name__ == "__main__":
    # === ILLUSTRATIVE numbers from the book (bioreactor DO model, Chapter 15) ===
    dev_cost = DevelopmentCost(
        person_months=22,
        rate_per_person_month=25_000,
        one_time_infrastructure=150_000,
    )
    maint_cost = AnnualMaintenanceCost(
        monitoring_retraining=45_000,
        periodic_requalification=40_000,
        infrastructure=60_000,
    )
    benefits = AnnualBenefits(
        avoided_formal_deviations=2 * 60_000,   # 2/year x 60k PLN
        avoided_batch_loss=800_000 / 3,          # batch value / 3 years
    )

    result = calculate_roi(dev_cost, maint_cost, benefits, batches_per_year=24)

    print("=== Verification against the book (Chapter 15) ===\n")
    for key, value in result.items():
        print(f"{key:<30} {value:>15,.0f}")

    print(f"\nThe book states: payback ~2.9 years (34 months), net cost ~PLN 10,200/batch")
    print(f"Our calculator: payback {result['payback_years']:.1f} years ({result['payback_months']:.0f} months), net {-result['net_per_batch']:,.0f} PLN/batch")
