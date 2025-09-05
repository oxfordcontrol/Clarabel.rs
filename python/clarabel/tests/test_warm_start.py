"""
Test warm start functionality in Python bindings.
"""

import sys
import os

# Add the path to the clarabel python module if it's not in the system path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

try:
    import clarabel
    import numpy as np
    
    def test_warm_start_settings():
        """Test that warm start settings are properly exposed."""
        
        # Create default settings
        settings = clarabel.DefaultSettings()
        
        # Test that warm start fields exist and have correct default values
        assert hasattr(settings, 'warm_start_enable')
        assert hasattr(settings, 'warm_start_x')
        assert hasattr(settings, 'warm_start_s')
        assert hasattr(settings, 'warm_start_z')
        assert hasattr(settings, 'warm_start_tau')
        assert hasattr(settings, 'warm_start_kappa')
        
        # Test default values
        assert settings.warm_start_enable == False
        assert settings.warm_start_x is None
        assert settings.warm_start_s is None
        assert settings.warm_start_z is None
        assert settings.warm_start_tau is None
        assert settings.warm_start_kappa is None
        
        # Test setting values
        settings.warm_start_enable = True
        settings.warm_start_x = [1.0, 2.0, 3.0]
        settings.warm_start_s = [0.5, 1.5]
        settings.warm_start_z = [0.1, 0.2]
        settings.warm_start_tau = 1.0
        settings.warm_start_kappa = 0.5
        
        # Test getting values
        assert settings.warm_start_enable == True
        assert settings.warm_start_x == [1.0, 2.0, 3.0]
        assert settings.warm_start_s == [0.5, 1.5]
        assert settings.warm_start_z == [0.1, 0.2]
        assert settings.warm_start_tau == 1.0
        assert settings.warm_start_kappa == 0.5
        
        print("✓ All warm start settings tests passed!")
        return True
        
    def test_simple_warm_start():
        """Test basic warm start functionality."""
        
        # Simple QP problem with only inequality constraints
        # min x^2 + y^2 s.t. x >= 0.5, y >= 0.5, x <= 2, y <= 2  
        from scipy import sparse
        
        P = sparse.csc_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([0.0, 0.0])
        
        # Only inequality constraints to avoid negative dual variables
        A = sparse.csc_matrix([
            [1., 0.],    # x >= 0.5 becomes x - 0.5 >= 0, so slack s[0] = x - 0.5
            [0., 1.],    # y >= 0.5 becomes y - 0.5 >= 0, so slack s[1] = y - 0.5
            [-1., 0.],   # x <= 2 becomes -x + 2 >= 0, so slack s[2] = -x + 2
            [0., -1.]    # y <= 2 becomes -y + 2 >= 0, so slack s[3] = -y + 2
        ])
        b = np.array([0.5, 0.5, 2.0, 2.0])
        cones = [clarabel.NonnegativeConeT(4)]
        
        # First solve
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        sol1 = solver1.solve()
        
        # Second solve with warm start
        settings_warm = clarabel.DefaultSettings()
        settings_warm.verbose = False
        settings_warm.warm_start_enable = True
        settings_warm.warm_start_x = sol1.x
        settings_warm.warm_start_s = sol1.s
        settings_warm.warm_start_z = sol1.z
        settings_warm.warm_start_tau = sol1.tau
        settings_warm.warm_start_kappa = sol1.kappa
        
        # Slightly modified problem
        q_new = np.array([0.01, 0.01])
        solver2 = clarabel.DefaultSolver(P, q_new, A, b, cones, settings_warm)
        sol2 = solver2.solve()
        
        # Check that both solved successfully
        assert sol1.status == clarabel.SolverStatus.Solved
        assert sol2.status == clarabel.SolverStatus.Solved
        
        # Check warm start status
        assert sol1.warm_start_used == False  # First solve without warm start
        assert sol2.warm_start_used == True   # Second solve should use warm start
        
        print("✓ Basic warm start functionality test passed!")
        print(f"  Cold start iterations: {sol1.iterations}")
        print(f"  Warm start iterations: {sol2.iterations}")
        print(f"  Warm start used: {sol2.warm_start_used}")
        return True
    
    if __name__ == "__main__":
        print("Testing Clarabel Python warm start bindings...")
        
        try:
            test_warm_start_settings()
            test_simple_warm_start()
            print("\n🎉 All tests passed! Warm start functionality is properly exposed in Python.")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            sys.exit(1)
            
except ImportError as e:
    print(f"Could not import required modules: {e}")
    print("This is expected if clarabel Python bindings are not built/installed.")
    print("The bindings should work once the project is properly built.")
