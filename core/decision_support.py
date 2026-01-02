class DecisionSupport:
    def __init__(self, unit_capacities):
        """
        unit_capacities: List of MW capacities, e.g., [300, 250, 200]
        """
        # Sort capacities descending for better commitment logic (largest units first)
        self.unit_capacities = sorted(unit_capacities, reverse=True)
        self.units = [f"Unit {i+1}" for i in range(len(unit_capacities))]

    def recommend_units(self, predicted_load):
        """
        Simple greedy approach for unit commitment.
        Returns a list of ON units and OFF units.
        """
        on_units = []
        off_units = []
        current_capacity = 0
        
        # Add a 10% spinning reserve margin
        required_load = predicted_load * 1.1 
        
        for i, capacity in enumerate(self.unit_capacities):
            if current_capacity < required_load:
                on_units.append(f"Unit {i+1} ({capacity} MW)")
                current_capacity += capacity
            else:
                off_units.append(f"Unit {i+1} ({capacity} MW)")
                
        return on_units, off_units, current_capacity

    def identify_maintenance_windows(self, predictions, threshold_percentile=20):
        """
        Identify periods where load is in the bottom Nth percentile.
        """
        import numpy as np
        threshold = np.percentile(predictions, threshold_percentile)
        maintenance_indices = np.where(predictions <= threshold)[0]
        return maintenance_indices.tolist(), threshold
