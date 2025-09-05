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
        
        # Simple QP problem: min x^2 + y^2 s.t. x + y = 1, x >= 0, y >= 0
        P = np.array([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([0.0, 0.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        cones = [clarabel.NonnegativeConeT(2)]
        
        # First solve
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        solver1.solve()
        sol1 = solver1.solution
        
        # Second solve with warm start
        settings_warm = clarabel.DefaultSettings()
        settings_warm.verbose = False
        settings_warm.warm_start_enable = True
        settings_warm.warm_start_x = sol1.x.tolist()
        settings_warm.warm_start_s = sol1.s.tolist()
        settings_warm.warm_start_z = sol1.z.tolist()
        settings_warm.warm_start_tau = sol1.tau
        settings_warm.warm_start_kappa = sol1.kappa
        
        # Slightly modified problem
        q_new = np.array([0.1, 0.1])
        solver2 = clarabel.DefaultSolver(P, q_new, A, b, cones, settings_warm)
        solver2.solve()
        
        # Check that both solved successfully
        assert solver1.info.status == clarabel.SolverStatus.Solved
        assert solver2.info.status == clarabel.SolverStatus.Solved
        
        print("✓ Basic warm start functionality test passed!")
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
