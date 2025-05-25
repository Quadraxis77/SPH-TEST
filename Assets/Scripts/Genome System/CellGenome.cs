using System;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "NewCellGenome", menuName = "Genome Editor/Cell Genome", order = 1)]
public class CellGenome : ScriptableObject
{
    public static event Action OnGenomeChanged;

    public List<GenomeMode> modes = new List<GenomeMode>();

    public void RefreshModeIndexes()
    {
        for (int i = 0; i < modes.Count; i++)
        {
            modes[i].index = i;
            if (string.IsNullOrEmpty(modes[i].modeName))
                modes[i].modeName = $"Mode {i}";
        }
    }

    /// <summary>
    /// Checks if more than one mode is marked as initial and returns a list of all initial modes.
    /// </summary>
    /// <returns>A list of indices for all modes marked as initial</returns>
    public List<int> GetInitialModes()
    {
        List<int> initialModes = new List<int>();
        for (int i = 0; i < modes.Count; i++)
        {
            if (modes[i].isInitial)
            {
                initialModes.Add(i);
            }
        }
        return initialModes;
    }

    /// <summary>
    /// Ensures only one mode is marked as initial.
    /// If multiple modes are marked, only the first is kept.
    /// If none are marked, the first mode is set as initial.
    /// </summary>
    public void EnforceSingleInitialMode()
    {
        bool foundInitial = false;
        for (int i = 0; i < modes.Count; i++)
        {
            if (modes[i].isInitial)
            {
                if (!foundInitial)
                {
                    foundInitial = true;
                }
                else
                {
                    // More than one marked as initial; reset this one.
                    modes[i].isInitial = false;
                }
            }
        }
        // If none are marked as initial and there is at least one mode, mark the first one.
        if (!foundInitial && modes.Count > 0)
        {
            modes[0].isInitial = true;
        }
    }

    /// <summary>
    /// Validates that the genome configuration is valid for simulation.
    /// Throws an error if multiple initial modes are detected at runtime.
    /// </summary>
    public void ValidateForSimulation()
    {
        List<int> initialModes = GetInitialModes();
        
        if (initialModes.Count == 0 && modes.Count > 0)
        {
            // If no initial modes, set the first one
            modes[0].isInitial = true;
            Debug.LogWarning("No initial mode was set. Setting the first mode as initial.");
        }
        else if (initialModes.Count > 1)
        {
            // Multiple initial modes detected at runtime - throw an error
            string modeList = string.Join(", ", initialModes.ConvertAll(idx => $"'{modes[idx].modeName}'"));
            throw new System.InvalidOperationException(
                $"Multiple initial modes detected: {modeList}. Only one mode can be marked as initial during simulation.");
        }
    }    void OnValidate()
    {
        RefreshModeIndexes();

        // Don't enforce any initial mode requirements in the editor
        // This allows all modes, including the first one, to be unchecked

        if (Application.isPlaying)
        {
            Debug.Log("[OnValidate] Invoking OnGenomeChanged event.");
            // Add a small delay to ensure Unity has processed the changes
            UnityEditor.EditorApplication.delayCall += () => {
                OnGenomeChanged?.Invoke();
            };
        }
    }
    
    // Simplified method to manually trigger the event when needed
    public void TriggerGenomeChanged()
    {
        Debug.Log("[TriggerGenomeChanged] Manually triggering genome changed event");
        RefreshModeIndexes();
        
        // Only validate for simulation if we're in play mode
        if (Application.isPlaying)
        {
            ValidateForSimulation();
        }
        
        // Invoke the event directly
        OnGenomeChanged?.Invoke();
    }
}

[Serializable]
public class GenomeMode
{
    [HideInInspector] public int index;
    public string modeName;
    [Range(1f, 15f)] public float splitInterval = 5f;
    
    // New flag: only one mode should be marked as initial.
    [SerializeField]
    public bool isInitial = false;
    public bool parentMakeAdhesion = false;
    
    public Color modeColor = Color.white;

    [Header("Parent Split Settings")]
    [Range(-180f, 180f)] public float parentSplitYaw = 0f;
    [Range(-90f, 90f)] public float parentSplitPitch = 0f;

    [Header("Child A Settings")]
    // If set to -1, use the parent's mode.
    public int childAModeIndex = -1;
    [Range(-180f, 180f)] public float childA_OrientationYaw = 0f;
    [Range(-90f, 90f)] public float childA_OrientationPitch = 0f;
    public bool childA_KeepAdhesion = false;

    [Header("Child B Settings")]
    // If set to -1, use the parent's mode.
    public int childBModeIndex = -1;
    [Range(-180f, 180f)] public float childB_OrientationYaw = 0f;
    [Range(-90f, 90f)] public float childB_OrientationPitch = 0f;
    public bool childB_KeepAdhesion = false;

    [Header("Adhesion Settings")]
    [Range(1f, 10f)] public float adhesionRestLength = 3.0f;      // Natural length of the adhesion spring
    [Range(10f, 500f)] public float adhesionSpringStiffness = 100f; // Spring constant (10-500)
    [Range(0f, 100f)] public float adhesionSpringDamping = 5f;     // Damping coefficient (0-100)
    
    // Orientation constraints
    [Header("Orientation Constraints")]
    [Range(0f, 1f)] public float orientationConstraintStrength = 0.5f;
    [Range(0f, 180f)] public float maxAllowedAngleDeviation = 45f;
    
    // Make connections break under extreme force?
    [Header("Adhesion Breaking")]
    public bool adhesionCanBreak = false;
    [Range(100f, 5000f)] public float adhesionBreakForce = 1000f;
}
