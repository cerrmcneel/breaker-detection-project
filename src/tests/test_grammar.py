import pytest

def test_rail_overflow_protection():
    """
    Test that a Rail (Max 12 modules) correctly rejects 
    components that exceed its capacity.
    """
    from src.data_gen.grammar import Rail, Breaker
    
    # Initialize a standard 12-module rail
    rail = Rail(max_modules=12)
    
    # 1. Add a Main Breaker (4 modules)
    rail.add_component(Breaker(cls='MAINBREAKER', width=4))
    
    # 2. Add 8 MCBs (8 modules)
    for _ in range(8):
        rail.add_component(Breaker(cls='MCB', width=1))
        
    # 3. Attempt to add one more MCB (Total would be 13)
    result = rail.add_component(Breaker(cls='MCB', width=1))
    
    assert result is False, "Rail allowed overflow beyond 12 modules!"
    assert rail.current_width() == 12, "Rail width tracking is incorrect."

def test_panel_coordinate_generation():
    """
    Test that a Panel correctly calculates the Y-offset 
    for its rails based on a fixed rail_height.
    """
    from src.data_gen.grammar import Panel
    
    # Initialize a 3-rail panel with a 200px vertical gap between rails
    panel = Panel(rails_count=3, rail_height=200)
    
    # We expect the Y-coordinates of the rail centers to be:
    # Rail 0: 100px (Center of 0-200)
    # Rail 1: 300px (Center of 200-400)
    # Rail 2: 500px (Center of 400-600)
    
    assert panel.get_rail_y_center(0) == 100
    assert panel.get_rail_y_center(1) == 300
    assert panel.get_rail_y_center(2) == 500
