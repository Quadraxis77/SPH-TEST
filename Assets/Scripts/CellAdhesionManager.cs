using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class CellAdhesionManager : MonoBehaviour
{
    public ParticleSystemController particleSystemController;
    public Material bondMaterial;
    public float bondWidth = 0.05f;
    public Color zoneAColor = Color.green;
    public Color zoneBColor = Color.blue;
    public Color inheritanceZoneColor = Color.red;

    private List<AdhesionBond> bonds = new List<AdhesionBond>();
    private List<LineRenderer> bondLines = new List<LineRenderer>();

    // Metadata for each bond
    public class AdhesionBond
    {
        public int cellA;
        public int cellB;
        public BondZone zoneA; // zone for cellA's end
        public BondZone zoneB; // zone for cellB's end
        // Add more metadata as needed
    }

    public enum BondZone { ZoneA, ZoneB, Inheritance }

    void Awake()
    {
        if (particleSystemController == null)
        {
            particleSystemController = Object.FindFirstObjectByType<ParticleSystemController>();
            Debug.LogWarning($"[CellAdhesionManager] particleSystemController was not assigned in inspector. Auto-assigned: {particleSystemController != null}");
        }
        if (bondMaterial == null)
        {
            bondMaterial = new Material(Shader.Find("Sprites/Default"));
            Debug.LogWarning("[CellAdhesionManager] bondMaterial was not assigned. Using default material.");
        }
    }

    void LateUpdate()
    {
        UpdateBondVisuals();
    }

    public void AddBond(int cellA, int cellB, BondZone zoneA, BondZone zoneB)
    {
        // Prevent duplicate bonds (regardless of order)
        if (cellA == cellB || cellA < 0 || cellB < 0) return;
        if (bonds.Exists(b => (b.cellA == cellA && b.cellB == cellB) || (b.cellA == cellB && b.cellB == cellA)))
            return;
        bonds.Add(new AdhesionBond
        {
            cellA = cellA,
            cellB = cellB,
            zoneA = zoneA,
            zoneB = zoneB
        });
    }

    public void ClearBonds()
    {
        bonds.Clear();
        foreach (var lr in bondLines)
        {
            if (lr != null) Destroy(lr.gameObject);
        }
        bondLines.Clear();
    }

    // Helper to map uniqueID to current array index
    private int GetIndexForUniqueID(int uniqueID)
    {
        if (particleSystemController == null || particleSystemController.ParticleIDs == null)
            return -1;
        var ids = particleSystemController.ParticleIDs;
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i].uniqueID == uniqueID)
                return i;
        }
        return -1;
    }

    private void UpdateBondVisuals()
    {
        if (particleSystemController == null)
        {
            Debug.LogError("[CellAdhesionManager] particleSystemController is null. Bonds cannot be visualized.");
            return;
        }
        UpdateBondZones();
        foreach (var lr in bondLines)
        {
            if (lr != null) Destroy(lr.gameObject);
        }
        bondLines.Clear();
        var positions = particleSystemController.CpuParticlePositions;
        foreach (var bond in bonds)
        {
            int idxA = GetIndexForUniqueID(bond.cellA);
            int idxB = GetIndexForUniqueID(bond.cellB);
            if (idxA < 0 || idxB < 0 || idxA >= positions.Length || idxB >= positions.Length)
                continue;
            Vector3 posA = positions[idxA];
            Vector3 posB = positions[idxB];
            Vector3 midpoint = (posA + posB) * 0.5f;
            Color colorA = bond.zoneA == BondZone.ZoneA ? zoneAColor : bond.zoneA == BondZone.ZoneB ? zoneBColor : inheritanceZoneColor;
            Color colorB = bond.zoneB == BondZone.ZoneA ? zoneAColor : bond.zoneB == BondZone.ZoneB ? zoneBColor : inheritanceZoneColor;
            var goA = new GameObject($"Bond_{bond.cellA}_to_mid");
            var lrA = goA.AddComponent<LineRenderer>();
            lrA.positionCount = 2;
            lrA.SetPosition(0, posA);
            lrA.SetPosition(1, midpoint);
            lrA.startWidth = bondWidth;
            lrA.endWidth = bondWidth;
            lrA.material = bondMaterial != null ? bondMaterial : new Material(Shader.Find("Sprites/Default"));
            lrA.startColor = colorA;
            lrA.endColor = colorA;
            bondLines.Add(lrA);
            var goB = new GameObject($"Bond_mid_to_{bond.cellB}");
            var lrB = goB.AddComponent<LineRenderer>();
            lrB.positionCount = 2;
            lrB.SetPosition(0, midpoint);
            lrB.SetPosition(1, posB);
            lrB.startWidth = bondWidth;
            lrB.endWidth = bondWidth;
            lrB.material = bondMaterial != null ? bondMaterial : new Material(Shader.Find("Sprites/Default"));
            lrB.startColor = colorB;
            lrB.endColor = colorB;
            bondLines.Add(lrB);
        }
    }

    public BondZone ClassifyBondDirection(Vector3 cellPos, Quaternion cellRot, Vector3 otherPos, float splitYaw, float splitPitch, float inheritanceAngleDeg = 10f)
    {
        Vector3 bondDirWorld = (otherPos - cellPos).normalized;
        Vector3 bondDirLocal = Quaternion.Inverse(cellRot) * bondDirWorld;
        Vector3 forwardLocal = Vector3.forward;
        Vector3 splitDirLocal = Quaternion.Euler(splitPitch, splitYaw, 0f) * forwardLocal;
        float dot = Vector3.Dot(bondDirLocal, splitDirLocal);
        float angle = Mathf.Acos(Mathf.Clamp(dot, -1f, 1f)) * Mathf.Rad2Deg;
        float halfWidth = inheritanceAngleDeg * 1f;
        float equatorialAngle = 90f;
        if (Mathf.Abs(angle - equatorialAngle) <= halfWidth)
            return BondZone.Inheritance;
        else if (dot > 0)
            return BondZone.ZoneB;
        else
            return BondZone.ZoneA;
    }

    private void UpdateBondZones()
    {
        if (particleSystemController == null || particleSystemController.ParticleIDs == null || particleSystemController.genome == null)
            return;
        var genome = particleSystemController.genome;
        var particleIDs = particleSystemController.ParticleIDs;
        var positions = particleSystemController.CpuParticlePositions;
        var rotations = particleSystemController.CpuParticleRotations; // Use public property now
        System.Func<int, int> getModeIndex = idx =>
        {
            var cached = typeof(ParticleSystemController)
                .GetField("cachedParticleData", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                ?.GetValue(particleSystemController) as System.Array;
            if (cached != null && idx < cached.Length)
            {
                var modeIndexField = cached.GetType().GetElementType().GetField("modeIndex");
                if (modeIndexField != null)
                    return (int)modeIndexField.GetValue(cached.GetValue(idx));
            }
            return 0;
        };
        foreach (var bond in bonds)
        {
            int idxA = GetIndexForUniqueID(bond.cellA);
            int idxB = GetIndexForUniqueID(bond.cellB);
            if (idxA < 0 || idxB < 0 || idxA >= positions.Length || idxB >= positions.Length ||
                rotations == null || idxA >= rotations.Length || idxB >= rotations.Length)
                continue;
            int modeA = getModeIndex(idxA);
            int modeB = getModeIndex(idxB);
            float splitYawA = 0f, splitPitchA = 0f;
            float splitYawB = 0f, splitPitchB = 0f;
            if (modeA >= 0 && modeA < genome.modes.Count)
            {
                splitYawA = genome.modes[modeA].parentSplitYaw;
                splitPitchA = genome.modes[modeA].parentSplitPitch;
            }
            if (modeB >= 0 && modeB < genome.modes.Count)
            {
                splitYawB = genome.modes[modeB].parentSplitYaw;
                splitPitchB = genome.modes[modeB].parentSplitPitch;
            }
            Vector3 posA = positions[idxA];
            Vector3 posB = positions[idxB];
            Quaternion rotA = rotations[idxA];
            Quaternion rotB = rotations[idxB];
            // Evaluate zone for A->B and B->A directions using actual rotations
            bond.zoneA = ClassifyBondDirection(posA, rotA, posB, splitYawA, splitPitchA);
            bond.zoneB = ClassifyBondDirection(posB, rotB, posA, splitYawB, splitPitchB);
        }
    }    // Called by ParticleSystemController after a cell splits
    public void HandleCellSplit(
        int parentIndex,
        int childAIndex,
        int childBIndex,
        float parentSplitYaw,
        float parentSplitPitch,
        Quaternion parentRotation,
        Vector3 parentPosition,
        int childAModeIndex,
        int childBModeIndex,
        bool parentMakeAdhesion,
        bool childA_KeepAdhesion,
        bool childB_KeepAdhesion)
    {
        if (particleSystemController == null || 
            childAIndex < 0 || childBIndex < 0 ||
            childAIndex >= particleSystemController.ParticleIDs.Length ||
            childBIndex >= particleSystemController.ParticleIDs.Length)
            return;
          // Get uniqueIDs for the new cells
        int uniqueA = particleSystemController.ParticleIDs[childAIndex].uniqueID;
        int uniqueB = particleSystemController.ParticleIDs[childBIndex].uniqueID;
        
        // Get the parent's uniqueID - use the parentID from the children's records
        // Since the parent is replaced by childA, the parentID of either child should be the uniqueID of their parent
        int parentUniqueID = particleSystemController.ParticleIDs[childAIndex].parentID;
        
        // Find all bonds that involve the parent cell and transfer them to the child cells
        List<AdhesionBond> parentBonds = new List<AdhesionBond>();
        
        // Collect all bonds that involved the parent (will be modified, so we need a separate collection)
        foreach (var bond in bonds.ToList())
        {
            // Check if this bond involves the parent cell
            if (bond.cellA == parentUniqueID || bond.cellB == parentUniqueID)
            {
                parentBonds.Add(bond);
                
                // Optionally remove the parent's bond since the parent is now split
                bonds.Remove(bond);
            }
        }
          // Create new bonds between the children and the parent's neighbors
        foreach (var parentBond in parentBonds)
        {
            int neighborID = (parentBond.cellA == parentUniqueID) ? parentBond.cellB : parentBond.cellA;
            BondZone neighborZone = (parentBond.cellA == parentUniqueID) ? parentBond.zoneB : parentBond.zoneA;
            BondZone parentZone = (parentBond.cellA == parentUniqueID) ? parentBond.zoneA : parentBond.zoneB;            // If the bond is in the inheritance zone, decide which child gets it or if it should be split
            if (parentZone == BondZone.Inheritance)
            {
                // Both children want to keep the adhesion - both get a bond to the neighbor
                if (childA_KeepAdhesion && childB_KeepAdhesion)
                {
                    AddBond(uniqueA, neighborID, BondZone.Inheritance, neighborZone);
                    AddBond(uniqueB, neighborID, BondZone.Inheritance, neighborZone);
                }
                // Only Child A wants to keep the adhesion
                else if (childA_KeepAdhesion)
                {
                    AddBond(uniqueA, neighborID, BondZone.Inheritance, neighborZone);
                }
                // Only Child B wants to keep the adhesion
                else if (childB_KeepAdhesion)
                {
                    AddBond(uniqueB, neighborID, BondZone.Inheritance, neighborZone);
                }
            }
            // If the bond is in Zone A, give it to Child A (if it keeps adhesion)
            else if (parentZone == BondZone.ZoneA && childA_KeepAdhesion)
            {
                AddBond(uniqueA, neighborID, BondZone.Inheritance, neighborZone);
            }
            // If the bond is in Zone B, give it to Child B (if it keeps adhesion)
            else if (parentZone == BondZone.ZoneB && childB_KeepAdhesion)
            {
                AddBond(uniqueB, neighborID, BondZone.Inheritance, neighborZone);
            }
        }
        
        // If parentMakeAdhesion is true, also create a bond between the two children
        if (parentMakeAdhesion)
        {
            AddBond(uniqueA, uniqueB, BondZone.Inheritance, BondZone.Inheritance);
        }
    }
}
