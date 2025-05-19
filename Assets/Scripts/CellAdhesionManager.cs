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
    public Color zoneCColor = Color.red;

    private List<AdhesionBond> bonds = new List<AdhesionBond>();
    private List<LineRenderer> bondLines = new List<LineRenderer>();

    // Metadata for each bond
    public class AdhesionBond
    {
        public int cellA;
        public int cellB;
        public BondZone zoneA; // zone for cellA's end
        public BondZone zoneB; // zone for cellB's end
        // Track if this bond is a direct Child-to-Child bond (exempt until either splits)
        public bool isChildToChild = false;
        // Store the unique IDs of the two children at bond creation
        public int childAUniqueID;
        public int childBUniqueID;
        // Track the initial zone configuration when the bond was created
        public BondZone initialZoneA;
        public BondZone initialZoneB;
        public int creationFrame; // Track the frame when bond was created
        // Add more metadata as needed
    }

    public enum BondZone { ZoneA, ZoneB, ZoneC }

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

    // Helper class to hold LineRenderers for a bond
    private class BondVisual
    {
        public LineRenderer lrA;
        public LineRenderer lrB;
    }
    // Map bonds to visuals
    private Dictionary<AdhesionBond, BondVisual> bondVisuals = new Dictionary<AdhesionBond, BondVisual>();

    public void AddBond(int cellA, int cellB, BondZone zoneA, BondZone zoneB, bool isChildToChild = false, int childAUniqueID = -1, int childBUniqueID = -1)
    {
        // Prevent duplicate bonds (regardless of order)
        if (cellA == cellB || cellA < 0 || cellB < 0) return;
        if (bonds.Exists(b => (b.cellA == cellA && b.cellB == cellB) || (b.cellA == cellB && b.cellB == cellA)))
            return;
        var bond = new AdhesionBond
        {
            cellA = cellA,
            cellB = cellB,
            zoneA = zoneA,
            zoneB = zoneB,
            initialZoneA = zoneA, // Store initial zone configuration 
            initialZoneB = zoneB,
            isChildToChild = isChildToChild,
            childAUniqueID = childAUniqueID,
            childBUniqueID = childBUniqueID,
            creationFrame = Time.frameCount // Set creation frame
        };
        bonds.Add(bond);
        // Create visuals for this bond
        CreateBondVisual(bond);
    }

    public void ClearBonds()
    {
        bonds.Clear();
        foreach (var vis in bondVisuals.Values)
        {
            if (vis.lrA != null) Destroy(vis.lrA.gameObject);
            if (vis.lrB != null) Destroy(vis.lrB.gameObject);
        }
        bondVisuals.Clear();
        bondLines.Clear();
    }

    private void CreateBondVisual(AdhesionBond bond)
    {
        // Create two LineRenderers for the bond
        var goA = new GameObject($"Bond_{bond.cellA}_to_mid");
        var lrA = goA.AddComponent<LineRenderer>();
        lrA.positionCount = 2;
        lrA.startWidth = bondWidth;
        lrA.endWidth = bondWidth;
        lrA.material = bondMaterial != null ? bondMaterial : new Material(Shader.Find("Sprites/Default"));
        var goB = new GameObject($"Bond_mid_to_{bond.cellB}");
        var lrB = goB.AddComponent<LineRenderer>();
        lrB.positionCount = 2;
        lrB.startWidth = bondWidth;
        lrB.endWidth = bondWidth;
        lrB.material = bondMaterial != null ? bondMaterial : new Material(Shader.Find("Sprites/Default"));
        bondVisuals[bond] = new BondVisual { lrA = lrA, lrB = lrB };
        bondLines.Add(lrA);
        bondLines.Add(lrB);
    }

    private void RemoveBondVisual(AdhesionBond bond)
    {
        if (bondVisuals.TryGetValue(bond, out var vis))
        {
            if (vis.lrA != null) Destroy(vis.lrA.gameObject);
            if (vis.lrB != null) Destroy(vis.lrB.gameObject);
            bondVisuals.Remove(bond);
            bondLines.Remove(vis.lrA);
            bondLines.Remove(vis.lrB);
        }
    }

    private void FilterBonds()
    {
        if (bonds.Count == 0 || particleSystemController == null)
            return;
        var bondPositions = particleSystemController.CpuParticlePositions;
        var toRemove = new HashSet<AdhesionBond>();
        // For each bond end (cell, zone), find all bonds where this is an end, and if more than one, keep only shortest
        // 1. For cellA ends
        var groupA = bonds
            .Where(b => b.creationFrame < Time.frameCount)
            .GroupBy(b => (b.cellA, b.zoneA));
        foreach (var group in groupA)
        {
            // Skip filtering if any bond in the group is half in ZoneC and half in ZoneA or ZoneB
            if (group.Any(b => (b.zoneA == BondZone.ZoneC && (b.zoneB == BondZone.ZoneA || b.zoneB == BondZone.ZoneB)) ||
                               ((b.zoneA == BondZone.ZoneA || b.zoneA == BondZone.ZoneB) && b.zoneB == BondZone.ZoneC)))
                continue;
            if (group.Count() > 1)
            {
                var shortest = group.OrderBy(b =>
                {
                    int idxA = GetIndexForUniqueID(b.cellA);
                    int idxB = GetIndexForUniqueID(b.cellB);
                    if (idxA < 0 || idxB < 0 || idxA >= bondPositions.Length || idxB >= bondPositions.Length) return float.MaxValue;
                    return Vector3.Distance(bondPositions[idxA], bondPositions[idxB]);
                }).First();
                foreach (var b in group)
                    if (b != shortest) toRemove.Add(b);
            }
        }
        // 2. For cellB ends
        var groupB = bonds
            .Where(b => b.creationFrame < Time.frameCount)
            .GroupBy(b => (b.cellB, b.zoneB));
        foreach (var group in groupB)
        {
            // Skip filtering if any bond in the group is half in ZoneC and half in ZoneA or ZoneB
            if (group.Any(b => (b.zoneB == BondZone.ZoneC && (b.zoneA == BondZone.ZoneA || b.zoneA == BondZone.ZoneB)) ||
                               ((b.zoneB == BondZone.ZoneA || b.zoneB == BondZone.ZoneB) && b.zoneA == BondZone.ZoneC)))
                continue;
            if (group.Count() > 1)
            {
                var shortest = group.OrderBy(b =>
                {
                    int idxA = GetIndexForUniqueID(b.cellA);
                    int idxB = GetIndexForUniqueID(b.cellB);
                    if (idxA < 0 || idxB < 0 || idxA >= bondPositions.Length || idxB >= bondPositions.Length) return float.MaxValue;
                    return Vector3.Distance(bondPositions[idxA], bondPositions[idxB]);
                }).First();
                foreach (var b in group)
                    if (b != shortest) toRemove.Add(b);
            }
        }
        if (toRemove.Count > 0)
        {
            foreach (var b in toRemove)
                RemoveBondVisual(b);
            bonds.RemoveAll(b => toRemove.Contains(b));
        }
    }

    private void UpdateBondVisuals()
    {
        if (particleSystemController == null)
        {
            Debug.LogError("[CellAdhesionManager] particleSystemController is null. Bonds cannot be visualized.");
            return;
        }
        UpdateBondZones();
        FilterBonds();
        // Remove visuals for bonds that no longer exist
        var bondsToRemove = bondVisuals.Keys.Except(bonds).ToList();
        foreach (var bond in bondsToRemove)
            RemoveBondVisual(bond);
        // Add visuals for new bonds
        foreach (var bond in bonds)
        {
            if (!bondVisuals.ContainsKey(bond))
                CreateBondVisual(bond);
        }
        var positions = particleSystemController.CpuParticlePositions;
        // Update positions/colors for all visuals
        foreach (var bond in bonds)
        {
            int idxA = GetIndexForUniqueID(bond.cellA);
            int idxB = GetIndexForUniqueID(bond.cellB);
            if (idxA < 0 || idxB < 0 || idxA >= positions.Length || idxB >= positions.Length)
                continue;
            Vector3 posA = positions[idxA];
            Vector3 posB = positions[idxB];
            Vector3 midpoint = (posA + posB) * 0.5f;
            Color colorA = bond.zoneA == BondZone.ZoneB ? zoneAColor : bond.zoneA == BondZone.ZoneA ? zoneBColor : zoneCColor;
            Color colorB = bond.zoneB == BondZone.ZoneB ? zoneAColor : bond.zoneB == BondZone.ZoneA ? zoneBColor : zoneCColor;
            var vis = bondVisuals[bond];
            vis.lrA.SetPosition(0, posA);
            vis.lrA.SetPosition(1, midpoint);
            vis.lrA.startColor = colorA;
            vis.lrA.endColor = colorA;
            vis.lrB.SetPosition(0, midpoint);
            vis.lrB.SetPosition(1, posB);
            vis.lrB.startColor = colorB;
            vis.lrB.endColor = colorB;
        }
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
            return BondZone.ZoneC;
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
            // Skip updating bonds more than 2 frames after creation
            if (Time.frameCount > bond.creationFrame + 2)
                continue;
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
            if (parentZone == BondZone.ZoneC)
            {                // Both children want to keep the adhesion - both get a bond to the neighbor
                if (childA_KeepAdhesion && childB_KeepAdhesion)
                {
                    AddBond(uniqueA, neighborID, parentBond.zoneA, neighborZone);
                    AddBond(uniqueB, neighborID, parentBond.zoneA, neighborZone);
                }
                // Only Child A wants to keep the adhesion
                else if (childA_KeepAdhesion)
                {
                    AddBond(uniqueA, neighborID, parentBond.zoneA, neighborZone);
                }
                // Only Child B wants to keep the adhesion
                else if (childB_KeepAdhesion)
                {
                    AddBond(uniqueB, neighborID, parentBond.zoneA, neighborZone);
                }
            }            // If the bond is in Zone A, give it to Child A (if it keeps adhesion)
            else if (parentZone == BondZone.ZoneB && childA_KeepAdhesion)
            {
                // Keep the same zone type when transferring to children
                AddBond(uniqueA, neighborID, BondZone.ZoneB, neighborZone);
            }
            // If the bond is in Zone B, give it to Child B (if it keeps adhesion)
            else if (parentZone == BondZone.ZoneA && childB_KeepAdhesion)
            {
                // Keep the same zone type when transferring to children
                AddBond(uniqueB, neighborID, BondZone.ZoneA, neighborZone);
            }
        }
        
        // If parentMakeAdhesion is true, also create a bond between the two children
        if (parentMakeAdhesion)
        {
            // Mark as direct child-to-child bond, store unique IDs for exemption
            AddBond(uniqueA, uniqueB, BondZone.ZoneC, BondZone.ZoneC, true, uniqueA, uniqueB);
        }
    }
}
