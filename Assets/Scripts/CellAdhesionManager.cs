using System.Collections.Generic;
using UnityEngine;

public class CellAdhesionManager : MonoBehaviour
{
    public ParticleSystemController particleSystemController;
    public Material bondMaterial;
    public float bondWidth = 0.05f;
    public Color defaultBondColor = new Color(1.0f, 0.7f, 0.8f, 1.0f); // Light pink
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
        public BondZone zone;
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

    public void AddBond(int cellA, int cellB, BondZone zone, bool isInherited, bool keepA, bool keepB, bool makeAdhesion)
    {
        Debug.Log($"AddBond called: {cellA} <-> {cellB}, zone={zone}, isInherited={isInherited}, keepA={keepA}, keepB={keepB}, makeAdhesion={makeAdhesion}");
        bonds.Add(new AdhesionBond
        {
            cellA = cellA,
            cellB = cellB,
            zone = zone,
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
    }

    private void UpdateBondVisuals()
    {
        Debug.Log($"UpdateBondVisuals called. Bonds count: {bonds.Count}");
        if (particleSystemController == null)
        {
            Debug.LogError("[CellAdhesionManager] particleSystemController is null. Bonds cannot be visualized.");
            return;
        }
        // Clean up old lines
        foreach (var lr in bondLines)
        {
            if (lr != null) Destroy(lr.gameObject);
        }
        bondLines.Clear();

        var positions = particleSystemController.CpuParticlePositions;
        int selected = particleSystemController.LastSelectedParticleID;

        foreach (var bond in bonds)
        {
            if (bond.cellA < 0 || bond.cellB < 0 || bond.cellA >= positions.Length || bond.cellB >= positions.Length)
                continue;
            var go = new GameObject($"Bond_{bond.cellA}_{bond.cellB}");
            var lr = go.AddComponent<LineRenderer>();
            lr.positionCount = 2;
            lr.SetPosition(0, positions[bond.cellA]);
            lr.SetPosition(1, positions[bond.cellB]);
            lr.startWidth = bondWidth;
            lr.endWidth = bondWidth;
            lr.material = bondMaterial != null ? bondMaterial : new Material(Shader.Find("Sprites/Default"));
            Color color = defaultBondColor;
            if (selected == bond.cellA || selected == bond.cellB)
            {
                switch (bond.zone)
                {
                    case BondZone.ZoneA: color = zoneAColor; break;
                    case BondZone.ZoneB: color = zoneBColor; break;
                    case BondZone.Inheritance: color = inheritanceZoneColor; break;
                }
            }
            lr.startColor = color;
            lr.endColor = color;
            bondLines.Add(lr);
        }
    }

    // Utility: Classify the zone between two cells
    public BondZone ClassifyZone(Vector3 parentPos, Quaternion parentRot, Vector3 otherPos, float splitYaw, float splitPitch)
    {
        // Calculate split plane normal
        Vector3 forward = parentRot * Vector3.forward;
        Vector3 up = parentRot * Vector3.up;
        Vector3 right = parentRot * Vector3.right;
        Vector3 splitDirLocal = Quaternion.Euler(splitPitch, splitYaw, 0f) * Vector3.forward;
        Vector3 splitNormal = right * splitDirLocal.x + up * splitDirLocal.y + forward * splitDirLocal.z;
        splitNormal.Normalize();
        // Vector from parent to other
        Vector3 toOther = (otherPos - parentPos).normalized;
        float dot = Vector3.Dot(toOther, splitNormal);
        float angle = Mathf.Acos(Mathf.Clamp(dot, -1f, 1f)) * Mathf.Rad2Deg;
        // Inheritance zone: within 10 degrees of the split plane
        if (Mathf.Abs(angle - 90f) <= 10f) return BondZone.Inheritance;
        // ZoneA: one hemisphere, ZoneB: the other
        return (dot > 0) ? BondZone.ZoneA : BondZone.ZoneB;
    }

    // Call this when a cell splits to update bonds
    public void HandleCellSplit(int parentIdx, int childAIdx, int childBIdx, float splitYaw, float splitPitch, Quaternion parentRot, Vector3 parentPos, int modeA, int modeB, bool makeAdhesion, bool keepA, bool keepB)
    {
        Debug.Log($"HandleCellSplit: parentIdx={parentIdx}, childAIdx={childAIdx}, childBIdx={childBIdx}, splitYaw={splitYaw}, splitPitch={splitPitch}, makeAdhesion={makeAdhesion}, keepA={keepA}, keepB={keepB}");
        // Remove and process all bonds involving the parent
        for (int i = bonds.Count - 1; i >= 0; i--)
        {
            var bond = bonds[i];
            bool involvesParent = (bond.cellA == parentIdx || bond.cellB == parentIdx);
            if (!involvesParent) continue;
            int other = (bond.cellA == parentIdx) ? bond.cellB : bond.cellA;
            BondZone zone = ClassifyZone(parentPos, parentRot, particleSystemController.CpuParticlePositions[other], splitYaw, splitPitch);
            // Inheritance zone: duplicate to both children if both keep, else to one child if only one keeps
            if (zone == BondZone.Inheritance)
            {
                if (keepA) AddBond(childAIdx, other, zone, true, keepA, keepB, makeAdhesion);
                if (keepB) AddBond(childBIdx, other, zone, true, keepA, keepB, makeAdhesion);
            }
            // ZoneA: assign to childA if keepA
            else if (zone == BondZone.ZoneA)
            {
                if (keepA) AddBond(childAIdx, other, zone, true, keepA, keepB, makeAdhesion);
            }
            // ZoneB: assign to childB if keepB
            else if (zone == BondZone.ZoneB)
            {
                if (keepB) AddBond(childBIdx, other, zone, true, keepA, keepB, makeAdhesion);
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
                    Debug.Log($"[Adhesion] Checking bond between {siblings[i]} and {siblings[j]}: exists={exists}");
                    if (!exists)
                    {
                        Debug.Log($"[Adhesion] Creating sibling bond: {siblings[i]} <-> {siblings[j]}");
                        AddBond(siblings[i], siblings[j], BondZone.Inheritance, false, keepA, keepB, true);
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
            {
                bool exists = bonds.Exists(b => (b.cellA == siblings[i] && b.cellB == siblings[j]) || (b.cellA == siblings[j] && b.cellB == siblings[i]));
                if (!exists)
                {
                    AddBond(siblings[i], siblings[j], BondZone.Inheritance, false, keepA, keepB, true);
                }
            }
        }
    }

    // TODO: Add logic for bond inheritance, duplication, and dropping during cell division
    // and for zone classification based on split plane and direction between cells.
}
