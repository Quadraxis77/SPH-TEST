using System.Collections.Generic;
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
        public bool isInherited;
        public bool keepA;
        public bool keepB;
        public bool makeAdhesion;
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
        // Removed test bond creation
    }

    void LateUpdate()
    {
        UpdateBondVisuals();
    }

    public void AddBond(int cellA, int cellB, BondZone zoneA, BondZone zoneB, bool isInherited, bool keepA, bool keepB, bool makeAdhesion)
    {
        Debug.Log($"AddBond called: {cellA} <-> {cellB}, zoneA={zoneA}, zoneB={zoneB}, isInherited={isInherited}, keepA={keepA}, keepB={keepB}, makeAdhesion={makeAdhesion}");
        bonds.Add(new AdhesionBond
        {
            cellA = cellA,
            cellB = cellB,
            zoneA = zoneA,
            zoneB = zoneB,
            isInherited = isInherited,
            keepA = keepA,
            keepB = keepB,
            makeAdhesion = makeAdhesion
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
    }    private void UpdateBondVisuals()
    {
        Debug.Log($"UpdateBondVisuals called. Bonds count: {bonds.Count}");
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
            if (bond.cellA < 0 || bond.cellB < 0 || bond.cellA >= positions.Length || bond.cellB >= positions.Length)
                continue;
                
            // Get the cell positions
            Vector3 posA = positions[bond.cellA];
            Vector3 posB = positions[bond.cellB];
            
            // Calculate the midpoint between the two cells
            Vector3 midpoint = (posA + posB) * 0.5f;
            
            // Get colors for each end based on zone
            Color colorA = bond.zoneA == BondZone.ZoneA ? zoneAColor : bond.zoneA == BondZone.ZoneB ? zoneBColor : inheritanceZoneColor;
            Color colorB = bond.zoneB == BondZone.ZoneA ? zoneAColor : bond.zoneB == BondZone.ZoneB ? zoneBColor : inheritanceZoneColor;
            
            // Create line from cellA to midpoint
            var goA = new GameObject($"Bond_{bond.cellA}_to_mid");
            var lrA = goA.AddComponent<LineRenderer>();
            lrA.positionCount = 2;
            lrA.SetPosition(0, posA);
            lrA.SetPosition(1, midpoint);
            lrA.startWidth = bondWidth;
            lrA.endWidth = bondWidth;
            lrA.material = bondMaterial != null ? bondMaterial : new Material(Shader.Find("Sprites/Default"));
            lrA.startColor = colorA;
            lrA.endColor = colorA;  // Same color for start and end to avoid blending
            bondLines.Add(lrA);
            
            // Create line from midpoint to cellB
            var goB = new GameObject($"Bond_mid_to_{bond.cellB}");
            var lrB = goB.AddComponent<LineRenderer>();
            lrB.positionCount = 2;
            lrB.SetPosition(0, midpoint);
            lrB.SetPosition(1, posB);
            lrB.startWidth = bondWidth;
            lrB.endWidth = bondWidth;
            lrB.material = bondMaterial != null ? bondMaterial : new Material(Shader.Find("Sprites/Default"));
            lrB.startColor = colorB;
            lrB.endColor = colorB;  // Same color for start and end to avoid blending
            bondLines.Add(lrB);
        }
    }    // Updated to define inheritance zone as a trench around the split equator
    public BondZone ClassifyBondDirection(Vector3 cellPos, Quaternion cellRot, Vector3 otherPos, float splitYaw, float splitPitch, float inheritanceAngleDeg = 10f)
    {
        // Calculate the bond direction in local coordinates relative to the cell
        Vector3 bondDirWorld = (otherPos - cellPos).normalized;
        Vector3 bondDirLocal = Quaternion.Inverse(cellRot) * bondDirWorld; // Transform bond direction to local space

        // Use the cell's forward vector as the reference direction
        Vector3 forwardLocal = Vector3.forward; // In local space, forward is always (0, 0, 1)

        // Calculate the split direction in local coordinates
        Vector3 splitDirLocal = Quaternion.Euler(splitPitch, splitYaw, 0f) * forwardLocal;

        // Calculate dot product to determine zone A or B
        float dot = Vector3.Dot(bondDirLocal, splitDirLocal);
        
        // Calculate the angle between the bond direction and the split direction
        float angle = Mathf.Acos(Mathf.Clamp(dot, -1f, 1f)) * Mathf.Rad2Deg;
        
        // Determine if the bond is near the equator (90 degrees from split direction)
        // The inheritance zone is now defined as angle between (90-halfWidth) and (90+halfWidth)
        float halfWidth = inheritanceAngleDeg * 0.5f;
        float equatorialAngle = 90f;
        
        // Check if bond is in the equatorial trench
        if (Mathf.Abs(angle - equatorialAngle) <= halfWidth)
            return BondZone.Inheritance;
        else if (dot > 0)
            return BondZone.ZoneB;
        else
            return BondZone.ZoneA;
    }

    // Call this when a cell splits to update bonds
    public void HandleCellSplit(int parentIdx, int childAIdx, int childBIdx, float splitYaw, float splitPitch, Quaternion parentRot, Vector3 parentPos, int modeA, int modeB, bool makeAdhesion, bool keepA, bool keepB)
    {
        Debug.Log($"HandleCellSplit: parentIdx={parentIdx}, childAIdx={childAIdx}, childBIdx={childBIdx}, splitYaw={splitYaw}, splitPitch={splitPitch}, makeAdhesion={makeAdhesion}, keepA={keepA}, keepB={keepB}");
        // Remove and process all bonds involving the parent
        for (int i = bonds.Count - 1; i >= 0; i--)
        {            var bond = bonds[i];
            bool involvesParent = (bond.cellA == parentIdx || bond.cellB == parentIdx);
            if (!involvesParent) continue;
            
            int other = (bond.cellA == parentIdx) ? bond.cellB : bond.cellA;
            Vector3 otherPos = particleSystemController.CpuParticlePositions[other];
              Debug.Log($"[Adhesion] Processing bond between parent {parentIdx} and {other}");
            Debug.Log($"[Adhesion] Split params: yaw={splitYaw}, pitch={splitPitch}, parent pos={parentPos}, other pos={otherPos}");
            Debug.Log($"[Adhesion] Parent rotation: {parentRot}");
            
            // Double check that we're processing the right parameters
            BondZone zoneA = ClassifyBondDirection(parentPos, parentRot, otherPos, splitYaw, splitPitch);
            BondZone zoneB = ClassifyBondDirection(otherPos, Quaternion.identity, parentPos, 0, 0); // For other, no split
            
            Debug.Log($"[Adhesion] Classified as zoneA: {zoneA}, zoneB: {zoneB}");
            
            // Assign to childA or childB based on zoneA
            if (zoneA == BondZone.ZoneA && keepA)
                AddBond(childAIdx, other, zoneA, zoneB, true, keepA, keepB, makeAdhesion);
            else if (zoneA == BondZone.ZoneB && keepB)
                AddBond(childBIdx, other, zoneA, zoneB, true, keepA, keepB, makeAdhesion);
            else if (zoneA == BondZone.Inheritance)
            {
                // Inheritance: assign to both children if desired
                if (keepA) AddBond(childAIdx, other, zoneA, zoneB, true, keepA, keepB, makeAdhesion);
                if (keepB) AddBond(childBIdx, other, zoneA, zoneB, true, keepA, keepB, makeAdhesion);
            }
            // Remove the old bond
            bonds.RemoveAt(i);
        }
        // If makeAdhesion is set, always create a bond between all siblings (not just the two children)
        if (makeAdhesion)
        {
            // Find all children of the same parent (including the new children)
            var siblings = new List<int>();
            int parentUniqueID = -1;
            if (particleSystemController != null && particleSystemController.ParticleIDs != null)
            {
                // Use the parentID of either child (should be the same)
                if (childAIdx < particleSystemController.ParticleIDs.Length)
                    parentUniqueID = particleSystemController.ParticleIDs[childAIdx].parentID;
                for (int i = 0; i < particleSystemController.ParticleIDs.Length; i++)
                {
                    if (particleSystemController.ParticleIDs[i].parentID == parentUniqueID)
                        siblings.Add(i);
                }
                // Ensure the two new children are included (in case not yet updated in ParticleIDs)
                if (!siblings.Contains(childAIdx)) siblings.Add(childAIdx);
                if (!siblings.Contains(childBIdx)) siblings.Add(childBIdx);
            }
            // Create a bond between every pair of siblings (if not already present)
            Debug.Log($"[Adhesion] Sibling list after split: [{string.Join(",", siblings)}]");
            for (int i = 0; i < siblings.Count; i++)
            {
                for (int j = i + 1; j < siblings.Count; j++)
                {
                    bool exists = bonds.Exists(b => (b.cellA == siblings[i] && b.cellB == siblings[j]) || (b.cellA == siblings[j] && b.cellB == siblings[i]));
                    Debug.Log($"[Adhesion] Checking bond between {siblings[i]} and {siblings[j]}: exists={exists}");                    if (!exists)
                    {
                        Debug.Log($"[Adhesion] Creating sibling bond: {siblings[i]} <-> {siblings[j]}");
                          // Use positions and a default orientation for classification
                        Vector3 posI = particleSystemController.CpuParticlePositions[siblings[i]];
                        Vector3 posJ = particleSystemController.CpuParticlePositions[siblings[j]];
                        
                        // Create a sensible orientation for classification
                        // Using a rotation looking from sibling i to sibling j
                        Vector3 dirItoJ = (posJ - posI).normalized;
                        Quaternion lookRotation = Quaternion.LookRotation(dirItoJ);
                        
                        Debug.Log($"[Adhesion Debug] Sibling bond classification - posI={posI}, posJ={posJ}, dirItoJ={dirItoJ}");
                        Debug.Log($"[Adhesion Debug] lookRotation={lookRotation}, splitYaw={splitYaw}, splitPitch={splitPitch}");
                        
                        // Use appropriate parameters to classify
                        BondZone zoneI = ClassifyBondDirection(posI, lookRotation, posJ, splitYaw, splitPitch);
                        BondZone zoneJ = ClassifyBondDirection(posJ, lookRotation, posI, splitYaw, splitPitch);
                        
                        Debug.Log($"[Adhesion] Classified sibling bond: {siblings[i]} <-> {siblings[j]} as zoneI={zoneI}, zoneJ={zoneJ}");
                        AddBond(siblings[i], siblings[j], zoneI, zoneJ, false, keepA, keepB, true);
                    }
                }
            }
        }
    }

    // Call this to create bonds between all initial cells if makeAdhesion is enabled
    public void CreateInitialSiblingBonds(bool makeAdhesion, bool keepA, bool keepB)
    {
        if (!makeAdhesion || particleSystemController == null || particleSystemController.ParticleIDs == null)
            return;
        var siblings = new List<int>();
        for (int i = 0; i < particleSystemController.ParticleIDs.Length; i++)
        {
            siblings.Add(i);
        }
        for (int i = 0; i < siblings.Count; i++)
        {
            for (int j = i + 1; j < siblings.Count; j++)
            {                bool exists = bonds.Exists(b => (b.cellA == siblings[i] && b.cellB == siblings[j]) || (b.cellA == siblings[j] && b.cellB == siblings[i]));
                if (!exists)
                {
                    // For initial bonds, randomly assign zone since we don't have a split direction
                    BondZone randomZoneA = Random.value > 0.5f ? BondZone.ZoneA : BondZone.ZoneB;
                    BondZone randomZoneB = Random.value > 0.5f ? BondZone.ZoneA : BondZone.ZoneB;
                    Debug.Log($"[Adhesion] Creating initial bond: {siblings[i]} <-> {siblings[j]} as zoneA={randomZoneA}, zoneB={randomZoneB}");
                    AddBond(siblings[i], siblings[j], randomZoneA, randomZoneB, false, keepA, keepB, true);
                }
            }
        }
    }    private void UpdateBondZones()
    {
        if (particleSystemController == null || particleSystemController.ParticleIDs == null || particleSystemController.genome == null)
            return;

        var genome = particleSystemController.genome;
        var particleIDs = particleSystemController.ParticleIDs;
        var positions = particleSystemController.CpuParticlePositions;
        // Try to get cached mode indices if available
        System.Func<int, int> getModeIndex = idx =>
        {
            // Try to get modeIndex from cached data if available
            var cached = typeof(ParticleSystemController)
                .GetField("cachedParticleData", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                ?.GetValue(particleSystemController) as System.Array;
            if (cached != null && idx < cached.Length)
            {
                var modeIndexField = cached.GetType().GetElementType().GetField("modeIndex");
                if (modeIndexField != null)
                    return (int)modeIndexField.GetValue(cached.GetValue(idx));
            }
            // Fallback: use 0
            return 0;
        };

        foreach (var bond in bonds)
        {
            int modeA = getModeIndex(bond.cellA);
            int modeB = getModeIndex(bond.cellB);
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
            Vector3 posA = positions[bond.cellA];
            Vector3 posB = positions[bond.cellB];
            bond.zoneA = ClassifyBondDirection(posA, Quaternion.identity, posB, splitYawA, splitPitchA);
            bond.zoneB = ClassifyBondDirection(posB, Quaternion.identity, posA, splitYawB, splitPitchB);
        }
    }
}
