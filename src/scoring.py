import pandas as pd

# Column names are single-sourced in src/data_schema.py (shared with the labeler that
# writes data/breaker_dataset.csv) so reader and writer cannot drift apart.
from src.data_schema import (
    COMMENTS,
    HAS_OVP,
    HAS_RCD,
    HAS_RCD_SI,
    PANEL_AGE,
    RCD_LOAD_RATIO,
    RCD_TEST_RESULT,
)


#define a function to calculate a safety score
def calculate_safety_score(row):
    score =50
    #overprotection bonus
    if row[HAS_OVP] == 1:
        score += 15
    #High immunity bonus, very rare in our dataset
    if row[HAS_RCD_SI] == 1:
        score += 15

    #add an age Penalty
    age = str(row[PANEL_AGE]).strip()
    if age =='> 20 years':
            score -= 15
            # RCDs and MCBs degrade overtime, it is not uncommon to find unresponsive RCDs
    elif age in ['10-15 years', '15 years']:
            score -= 5
    # Critical Failure (No RCD, this is the unit that protects people and can prevent other shorts from becoming a major problem.)
    if row[HAS_RCD] == 0:
            score -= 20
#rule of 5 Violation, rcd_load_ratio recommended no more than 5 per RCD
    if pd.notna(row[RCD_LOAD_RATIO]) and row[RCD_LOAD_RATIO] > 5:
            score -=10
    comments = str(row.get(COMMENTS, '')).lower()
    for kw in ['burnt', 'corroded', 'loose']:
        if kw in comments:
            score -= 15

    # RCD Test Penalties
    rcd_test = str(row.get(RCD_TEST_RESULT, '')).lower()
    if rcd_test == 'unresponsive':
        score -= 30
    elif rcd_test == 'slow':
        score -= 10

    return score
