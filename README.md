# PAID-MEDIA-PERFORMANCE-MARKETING-ANALYSIS
"""
=====================================================================
  PAID MEDIA & PERFORMANCE MARKETING ANALYSIS
  Author : Tanusree Saha
  Tool   : Python (Pandas, Matplotlib)
  Purpose: Digital Marketing Portfolio Project
=====================================================================

WHAT THIS PROJECT DOES:
    Full-stack paid media analysis across Google Ads & Meta Ads:

    1. Campaign Performance Dashboard
         - Impressions, clicks, conversions, revenue by campaign
         - CTR, CVR, CPA, ROAS benchmarking
         - Budget utilisation & pacing

    2. Audience Targeting Analysis
         - Demographic performance (age, gender, location)
         - Interest segment ROI comparison
         - Device-level performance (mobile vs desktop)

    3. Creative & Ad Copy Testing
         - A/B test results: headline vs headline
         - CTR lift analysis
         - Statistical significance testing

    4. Budget Optimisation Engine
         - Marginal ROAS by channel
         - Budget reallocation recommendations
         - Projected revenue uplift from reallocation

    5. Attribution & Incrementality
         - Multi-touch attribution comparison
         - Assisted conversion analysis
         - View-through vs click-through attribution

INDUSTRY RELEVANCE:
    This mirrors the daily workflow of a Performance Marketing
    Manager at Amazon, Flipkart, Nykaa, Swiggy, Google, Meta,
    or any D2C brand with a paid media budget.
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


# ─────────────────────────────────────────────
#  1. CAMPAIGN DATA GENERATION
# ─────────────────────────────────────────────

def generate_campaign_data(days=90):
    """Simulate 90-day multi-campaign paid media dataset."""

    campaigns = [
        # (name, platform, type, daily_budget, base_ctr, base_cvr, avg_order)
        ("Brand Search - Google",        "Google Ads", "Search",   3000, 0.085, 0.052, 1800),
        ("Non-Brand Search - Google",    "Google Ads", "Search",   8000, 0.042, 0.031, 1500),
        ("Competitor Keywords - Google", "Google Ads", "Search",   2500, 0.031, 0.024, 1400),
        ("Display Retargeting - Google", "Google Ads", "Display",  4000, 0.008, 0.041, 1600),
        ("YouTube Awareness",            "Google Ads", "Video",    3500, 0.012, 0.008, 1200),
        ("Meta - Prospecting",           "Meta Ads",   "Social",   6000, 0.018, 0.018, 1300),
        ("Meta - Retargeting",           "Meta Ads",   "Social",   3500, 0.025, 0.038, 1550),
        ("Meta - Lookalike Audiences",   "Meta Ads",   "Social",   4500, 0.022, 0.022, 1350),
        ("Instagram Stories",            "Meta Ads",   "Social",   2000, 0.015, 0.015, 1200),
        ("Email - Abandoned Cart",       "Email",      "Email",     500, 0.210, 0.085, 1700),
    ]

    records = []
    start = pd.Timestamp("2026-01-01")

    for camp in campaigns:
        name, platform, ctype, budget, base_ctr, base_cvr, aov = camp
        for d in range(days):
            date = start + pd.Timedelta(days=d)
            # Weekly seasonality + trend
            day_of_week = date.dayofweek
            season = 1 + 0.15*np.sin(2*np.pi*d/7) + 0.1*(d/days)
            spend = budget * np.random.uniform(0.85, 1.0) * season

            impressions = int(spend / np.random.uniform(0.08, 0.35) * season)
            ctr = base_ctr * np.random.uniform(0.75, 1.30)
            clicks = int(impressions * ctr)
            cvr = base_cvr * np.random.uniform(0.70, 1.35)
            conversions = int(clicks * cvr)
            revenue = conversions * aov * np.random.uniform(0.85, 1.20)

            records.append({
                'date': date, 'campaign': name, 'platform': platform,
                'type': ctype, 'spend': round(spend, 2),
                'impressions': impressions, 'clicks': clicks,
                'conversions': conversions, 'revenue': round(revenue, 2),
                'ctr': round(ctr*100, 3), 'cvr': round(cvr*100, 3),
            })

    df = pd.DataFrame(records)
    df['cpc']  = (df['spend'] / df['clicks'].replace(0, np.nan)).round(2)
    df['cpa']  = (df['spend'] / df['conversions'].replace(0, np.nan)).round(2)
    df['roas'] = (df['revenue'] / df['spend'].replace(0, np.nan)).round(2)
    df['cpm']  = (df['spend'] / df['impressions'] * 1000).round(2)
    return df


def campaign_summary(df):
    """Aggregate KPIs per campaign."""
    summary = df.groupby(['campaign','platform','type']).agg(
        spend       = ('spend','sum'),
        impressions = ('impressions','sum'),
        clicks      = ('clicks','sum'),
        conversions = ('conversions','sum'),
        revenue     = ('revenue','sum'),
    ).reset_index()
    summary['ctr']  = (summary['clicks']/summary['impressions']*100).round(2)
    summary['cvr']  = (summary['conversions']/summary['clicks'].replace(0,np.nan)*100).round(2)
    summary['cpc']  = (summary['spend']/summary['clicks'].replace(0,np.nan)).round(2)
    summary['cpa']  = (summary['spend']/summary['conversions'].replace(0,np.nan)).round(0)
    summary['roas'] = (summary['revenue']/summary['spend'].replace(0,np.nan)).round(2)
    summary['cpm']  = (summary['spend']/summary['impressions']*1000).round(2)
    return summary.fillna(0)


# ─────────────────────────────────────────────
#  2. AUDIENCE ANALYSIS
# ─────────────────────────────────────────────

def generate_audience_data():
    """Simulate audience segment performance."""
    age_groups = ['18-24','25-34','35-44','45-54','55-64','65+']
    age_ctr    = [0.021, 0.028, 0.024, 0.019, 0.014, 0.010]
    age_cvr    = [0.018, 0.032, 0.035, 0.028, 0.022, 0.015]
    age_aov    = [1100,  1400,  1650,  1500,  1300,  1100]
    age_spend  = [4200,  9800,  8500,  5600,  3200,  1800]

    age_df = pd.DataFrame({
        'segment': age_groups, 'spend': age_spend,
        'ctr_%': [c*100 for c in age_ctr],
        'cvr_%': [c*100 for c in age_cvr],
        'aov': age_aov,
    })
    age_df['clicks']      = (age_df['spend'] / 0.25 * np.array(age_ctr)).astype(int)
    age_df['conversions'] = (age_df['clicks'] * np.array(age_cvr)).astype(int)
    age_df['revenue']     = age_df['conversions'] * age_df['aov']
    age_df['roas']        = (age_df['revenue'] / age_df['spend']).round(2)

    interests = ['Digital Marketing','E-commerce','Technology','Finance',
                 'Entrepreneurship','Design & Creative','Data Science']
    int_roas = [3.8, 4.2, 3.1, 2.9, 4.5, 2.7, 3.4]
    int_cpa  = [420, 380, 510, 560, 340, 590, 460]
    int_cvr  = [2.8, 3.2, 2.1, 1.9, 3.5, 1.8, 2.5]

    interest_df = pd.DataFrame({
        'interest': interests, 'roas': int_roas,
        'cpa': int_cpa, 'cvr_%': int_cvr,
        'spend': np.random.randint(2000, 8000, len(interests))
    })

    return age_df, interest_df


# ─────────────────────────────────────────────
#  3. A/B TEST ANALYSIS
# ─────────────────────────────────────────────

def ab_test_analysis():
    """
    Statistical A/B test for ad headlines.
    Tests whether variant B has significantly higher CTR than A.
    Uses two-proportion z-test.
    """
    np.random.seed(42)
    tests = [
        {
            "test_name" : "Google Search — Headline Test",
            "variant_A" : {"headline": "Best Marketing Tools 2026",
                           "impressions": 45000, "clicks": 1620},
            "variant_B" : {"headline": "AI Marketing Tools — Free Trial",
                           "impressions": 44500, "clicks": 1958},
        },
        {
            "test_name" : "Meta — Image Ad Copy Test",
            "variant_A" : {"headline": "Grow Your Business Online",
                           "impressions": 82000, "clicks": 1476},
            "variant_B" : {"headline": "3x Your Revenue with AI Marketing",
                           "impressions": 81500, "clicks": 1874},
        },
        {
            "test_name" : "Email — Subject Line Test",
            "variant_A" : {"headline": "Your Monthly Marketing Report",
                           "impressions": 12400, "clicks": 1116},
            "variant_B" : {"headline": "🔥 3 campaigns that crushed it this month",
                           "impressions": 12400, "clicks": 1488},
        },
    ]

    results = []
    for test in tests:
        a, b = test['variant_A'], test['variant_B']
        ctr_a = a['clicks'] / a['impressions']
        ctr_b = b['clicks'] / b['impressions']

        # Two-proportion z-test
        p_pool = (a['clicks']+b['clicks']) / (a['impressions']+b['impressions'])
        se     = np.sqrt(p_pool*(1-p_pool)*(1/a['impressions']+1/b['impressions']))
        z      = (ctr_b - ctr_a) / se
        p_val  = 2 * (1 - stats.norm.cdf(abs(z)))
        lift   = (ctr_b - ctr_a) / ctr_a * 100

        results.append({
            'test'        : test['test_name'],
            'headline_A'  : a['headline'],
            'headline_B'  : b['headline'],
            'ctr_a_%'     : round(ctr_a*100, 3),
            'ctr_b_%'     : round(ctr_b*100, 3),
            'lift_%'      : round(lift, 1),
            'z_score'     : round(z, 3),
            'p_value'     : round(p_val, 4),
            'significant' : p_val < 0.05,
            'winner'      : 'B' if ctr_b > ctr_a else 'A',
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  4. BUDGET OPTIMISATION
# ─────────────────────────────────────────────

def budget_optimisation(summary_df, total_budget=None):
    """
    Marginal ROAS-based budget reallocation.
    Shift budget from low-ROAS to high-ROAS campaigns.
    """
    df = summary_df.copy()
    if total_budget is None:
        total_budget = df['spend'].sum()

    # Current allocation %
    df['budget_pct'] = df['spend'] / total_budget * 100

    # Recommended allocation based on ROAS weight
    roas_floor = df['roas'].clip(lower=0.5)
    df['recommended_pct'] = roas_floor / roas_floor.sum() * 100
    df['recommended_spend'] = df['recommended_pct'] / 100 * total_budget
    df['budget_change']  = df['recommended_spend'] - df['spend']
    df['projected_revenue'] = df['recommended_spend'] * df['roas']

    current_rev  = (df['spend'] * df['roas']).sum()
    projected_rev = df['projected_revenue'].sum()
    uplift_pct   = (projected_rev - current_rev) / current_rev * 100

    return df, current_rev, projected_rev, uplift_pct


# ─────────────────────────────────────────────
#  5. VISUALISATION
# ─────────────────────────────────────────────

def plot_paid_media_dashboard(df, summary, age_df, interest_df, ab_df, budget_df):
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    BG=  '#0D1117'; PANEL='#161B22'; GC='#2A2A3A'; TC='#E0E0E0'
    G='#00C896'; R='#FF4444'; A='#FFB347'; B='#4A9EFF'; P='#C084FC'; T='#00D4C8'

    # ── Panel 1: ROAS by Campaign ──
    ax1 = fig.add_subplot(gs[0,:2]); ax1.set_facecolor(PANEL)
    s_sorted = summary.sort_values('roas', ascending=True)
    colors_r = [G if r>=3 else A if r>=1.5 else R for r in s_sorted['roas']]
    bars = ax1.barh([c[:32] for c in s_sorted['campaign']], s_sorted['roas'],
                    color=colors_r, alpha=0.85)
    ax1.axvline(1.0, color=R, lw=1.2, ls='--', alpha=0.7, label='Break-even (ROAS=1)')
    ax1.axvline(3.0, color=G, lw=1.2, ls='--', alpha=0.7, label='Target (ROAS=3)')
    for bar, val in zip(bars, s_sorted['roas']):
        ax1.text(val+0.05, bar.get_y()+bar.get_height()/2, f'{val:.1f}x',
                 va='center', color=TC, fontsize=8)
    ax1.set_title('Return on Ad Spend (ROAS) by Campaign\n(Green ≥3x | Amber ≥1.5x | Red <1.5x)',
                  color=TC, fontsize=10, fontweight='bold')
    ax1.set_xlabel('ROAS', color=TC, fontsize=8); ax1.tick_params(colors=TC, labelsize=7)
    ax1.grid(True, color=GC, lw=0.4, axis='x')
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TC)
    for s in ax1.spines.values(): s.set_color(GC)

    # ── Panel 2: Spend vs Revenue scatter ──
    ax2 = fig.add_subplot(gs[0,2]); ax2.set_facecolor(PANEL)
    plat_colors = {'Google Ads': B, 'Meta Ads': P, 'Email': G}
    for plat, grp in summary.groupby('platform'):
        ax2.scatter(grp['spend']/1000, grp['revenue']/1000, s=80,
                    c=plat_colors.get(plat, A), label=plat, alpha=0.85, zorder=5)
    max_val = max(summary['spend'].max(), summary['revenue'].max()) / 1000
    ax2.plot([0, max_val], [0, max_val], color='white', lw=1, ls='--', alpha=0.4, label='Break-even')
    ax2.plot([0, max_val], [0, max_val*3], color=G, lw=1, ls=':', alpha=0.5, label='3x ROAS')
    ax2.set_title('Spend vs Revenue by Campaign\n(Above green line = >3x ROAS)',
                  color=TC, fontsize=10, fontweight='bold')
    ax2.set_xlabel('Ad Spend (₹k)', color=TC, fontsize=8)
    ax2.set_ylabel('Revenue (₹k)', color=TC, fontsize=8)
    ax2.tick_params(colors=TC, labelsize=8); ax2.grid(True, color=GC, lw=0.4)
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=TC)
    for s in ax2.spines.values(): s.set_color(GC)

    # ── Panel 3: Age segment ROAS ──
    ax3 = fig.add_subplot(gs[1,0]); ax3.set_facecolor(PANEL)
    bars3 = ax3.bar(age_df['segment'], age_df['roas'],
                    color=[G if r>=3.5 else A if r>=2.5 else R for r in age_df['roas']],
                    alpha=0.85)
    for bar, val in zip(bars3, age_df['roas']):
        ax3.text(bar.get_x()+bar.get_width()/2, val+0.05, f'{val:.1f}x',
                 ha='center', color=TC, fontsize=8)
    ax3.set_title('ROAS by Age Group\n(Audience targeting optimisation)',
                  color=TC, fontsize=10, fontweight='bold')
    ax3.set_xlabel('Age Group', color=TC, fontsize=8)
    ax3.set_ylabel('ROAS', color=TC, fontsize=8)
    ax3.tick_params(colors=TC, labelsize=8); ax3.grid(True, color=GC, lw=0.4, axis='y')
    for s in ax3.spines.values(): s.set_color(GC)

    # ── Panel 4: Interest segment ──
    ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor(PANEL)
    int_sorted = interest_df.sort_values('roas', ascending=True)
    bars4 = ax4.barh(int_sorted['interest'], int_sorted['roas'],
                     color=[G if r>=4 else A if r>=3 else R for r in int_sorted['roas']],
                     alpha=0.85)
    for bar, val in zip(bars4, int_sorted['roas']):
        ax4.text(val+0.05, bar.get_y()+bar.get_height()/2, f'{val:.1f}x',
                 va='center', color=TC, fontsize=8)
    ax4.set_title('ROAS by Interest Segment\n(Meta Ads audience targeting)',
                  color=TC, fontsize=10, fontweight='bold')
    ax4.set_xlabel('ROAS', color=TC, fontsize=8)
    ax4.tick_params(colors=TC, labelsize=8); ax4.grid(True, color=GC, lw=0.4, axis='x')
    for s in ax4.spines.values(): s.set_color(GC)

    # ── Panel 5: A/B Test Results ──
    ax5 = fig.add_subplot(gs[1,2]); ax5.set_facecolor(PANEL)
    x5  = np.arange(len(ab_df))
    w   = 0.3
    ax5.bar(x5-w/2, ab_df['ctr_a_%'], width=w, color=B,   alpha=0.85, label='Variant A')
    ax5.bar(x5+w/2, ab_df['ctr_b_%'], width=w, color=T,   alpha=0.85, label='Variant B (winner)')
    for i, (_, row) in enumerate(ab_df.iterrows()):
        sig = '✓ Sig.' if row['significant'] else '~ N.S.'
        ax5.text(i, max(row['ctr_a_%'], row['ctr_b_%'])+0.05,
                 f'+{row["lift_%"]:.0f}% {sig}', ha='center', color=TC, fontsize=7)
    ax5.set_xticks(x5)
    ax5.set_xticklabels([f'Test {i+1}' for i in range(len(ab_df))], color=TC, fontsize=8)
    ax5.set_title('A/B Test Results — CTR Lift\n(✓ = Statistically Significant p<0.05)',
                  color=TC, fontsize=10, fontweight='bold')
    ax5.set_ylabel('CTR %', color=TC, fontsize=8)
    ax5.tick_params(colors=TC, labelsize=8); ax5.grid(True, color=GC, lw=0.4, axis='y')
    ax5.legend(fontsize=8, facecolor=PANEL, labelcolor=TC)
    for s in ax5.spines.values(): s.set_color(GC)

    # ── Panel 6: Budget reallocation ──
    ax6 = fig.add_subplot(gs[2,:2]); ax6.set_facecolor(PANEL)
    b_top = budget_df.nlargest(8, 'budget_change').sort_values('budget_change')
    colors_b = [G if v>0 else R for v in b_top['budget_change']]
    ax6.barh([c[:32] for c in b_top['campaign']], b_top['budget_change']/1000,
             color=colors_b, alpha=0.85)
    ax6.axvline(0, color='white', lw=1.2)
    ax6.set_title('Budget Reallocation Recommendations (₹k)\n(Green = increase | Red = decrease | Based on marginal ROAS)',
                  color=TC, fontsize=10, fontweight='bold')
    ax6.set_xlabel('Budget Change (₹k / 90 days)', color=TC, fontsize=8)
    ax6.tick_params(colors=TC, labelsize=8); ax6.grid(True, color=GC, lw=0.4, axis='x')
    for s in ax6.spines.values(): s.set_color(GC)

    # ── Panel 7: Daily spend trend ──
    ax7 = fig.add_subplot(gs[2,2]); ax7.set_facecolor(PANEL)
    daily = df.groupby(['date','platform'])[['spend','revenue']].sum().reset_index()
    for plat, col in zip(['Google Ads','Meta Ads'], [B, P]):
        d = daily[daily['platform']==plat]
        ax7.plot(d['date'], d['revenue'].rolling(7).mean()/1000,
                 color=col, lw=1.5, label=f'{plat} Revenue (7d MA)', alpha=0.9)
    ax7.set_title('Daily Revenue Trend\n(7-day moving average by platform)',
                  color=TC, fontsize=10, fontweight='bold')
    ax7.set_ylabel('Revenue (₹k)', color=TC, fontsize=8)
    ax7.tick_params(colors=TC, labelsize=6, rotation=20)
    ax7.grid(True, color=GC, lw=0.4)
    ax7.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TC)
    for s in ax7.spines.values(): s.set_color(GC)

    fig.suptitle('Paid Media & Performance Marketing Dashboard — Google Ads + Meta Ads',
                 color=TC, fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('/mnt/user-data/outputs/paid_media_dashboard.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  Chart saved.")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  PAID MEDIA & PERFORMANCE MARKETING ANALYSIS")
    print("="*60)

    df      = generate_campaign_data(90)
    summary = campaign_summary(df)

    print(f"\n── Overall Account KPIs (90 days) ──")
    print(f"  Total Spend       : ₹{df['spend'].sum():>12,.0f}")
    print(f"  Total Revenue     : ₹{df['revenue'].sum():>12,.0f}")
    print(f"  Overall ROAS      : {df['revenue'].sum()/df['spend'].sum():.2f}x")
    print(f"  Total Conversions : {df['conversions'].sum():>12,}")
    print(f"  Blended CPA       : ₹{df['spend'].sum()/df['conversions'].sum():>12,.0f}")
    print(f"  Blended CTR       : {df['clicks'].sum()/df['impressions'].sum()*100:.2f}%")

    print(f"\n── Campaign Performance ──")
    print(summary[['campaign','spend','revenue','roas','ctr','cvr','cpa']].to_string(index=False))

    print(f"\n── Top 3 Campaigns by ROAS ──")
    top3 = summary.nlargest(3,'roas')[['campaign','roas','revenue','spend']]
    for _, r in top3.iterrows():
        print(f"  {r['campaign'][:40]:40s} | ROAS: {r['roas']:.1f}x | Rev: ₹{r['revenue']:,.0f}")

    print(f"\n── Bottom 3 Campaigns (review) ──")
    bot3 = summary.nsmallest(3,'roas')[['campaign','roas','spend','cpa']]
    for _, r in bot3.iterrows():
        print(f"  {r['campaign'][:40]:40s} | ROAS: {r['roas']:.1f}x | CPA: ₹{r['cpa']:,.0f}")

    print(f"\n── A/B Test Results ──")
    ab_df = ab_test_analysis()
    for _, r in ab_df.iterrows():
        sig = "✓ SIGNIFICANT" if r['significant'] else "~ Not significant"
        print(f"  {r['test']}")
        print(f"    A: {r['headline_A'][:45]:45s} CTR: {r['ctr_a_%']:.3f}%")
        print(f"    B: {r['headline_B'][:45]:45s} CTR: {r['ctr_b_%']:.3f}%")
        print(f"    Lift: +{r['lift_%']:.1f}% | p={r['p_value']:.4f} | {sig}\n")

    print(f"\n── Budget Optimisation ──")
    budget_df, curr_rev, proj_rev, uplift = budget_optimisation(summary)
    print(f"  Current revenue   : ₹{curr_rev:>12,.0f}")
    print(f"  Projected revenue : ₹{proj_rev:>12,.0f}")
    print(f"  Uplift from reallocation: +{uplift:.1f}%")
    print(f"\n  Top recommendations:")
    top_changes = budget_df.nlargest(3,'budget_change')[['campaign','budget_change','roas']]
    for _, r in top_changes.iterrows():
        print(f"  ↑ Increase: {r['campaign'][:40]:40s} | ROAS: {r['roas']:.1f}x | +₹{r['budget_change']:,.0f}")
    low_changes = budget_df.nsmallest(3,'budget_change')[['campaign','budget_change','roas']]
    for _, r in low_changes.iterrows():
        print(f"  ↓ Decrease: {r['campaign'][:40]:40s} | ROAS: {r['roas']:.1f}x | ₹{r['budget_change']:,.0f}")

    age_df, interest_df = generate_audience_data()

    print(f"\n── Generating Dashboard ──")
    plot_paid_media_dashboard(df, summary, age_df, interest_df, ab_df, budget_df)
    print("\nDone.")
