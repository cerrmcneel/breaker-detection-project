import pandas as pd
#define a function to calculate a safety score
def calculate_safety_score(row):
    score =50
    #overprotection bonus
    if row['has_ovp'] == 1:
        score += 15
    #High immunity bonus, very rare in our dataset
    if row['has_rcd_si'] == 1:
        score += 15

    #add an age Penalty
    age = str(row['panel_age']).strip()
    if age =='> 20 years': 
            score -= 15
            # RCDs and MCBs degrade overtime, it is not uncommon to find unresponsive RCDs
    elif age in ['10-15 years', '15 years']:
            score -= 5
    # Critical Failure (No RCD, this is the unit that protects people and can prevent other shorts from becoming a major problem.)
    if row['has_rcd'] == 0:
            score -= 20
#rule of 5 Violation, rcd_load_ratio recommended no more than 5 per RCD
    if pd.notna(row['rcd_load_ratio']) and row['rcd_load_ratio'] > 5:
            score -=10
    comments = str(row.get('comments', '')).lower()
    for kw in ['burnt', 'corroded', 'loose']:
        if kw in comments:
            score -= 15            
            
    # RCD Test Penalties
    rcd_test = str(row.get('rcd_test_result', '')).lower()
    if rcd_test == 'unresponsive':
        score -= 30
    elif rcd_test == 'slow':
        score -= 10
        
    return score

