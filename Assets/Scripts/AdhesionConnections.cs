using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class AdhesionConnections
{
    private static readonly float COS_CUTOFF = Mathf.Cos(Mathf.Deg2Rad * 92f); // cos(coneHalfAngle), half-angle of the adhesion cone

    // New formatted ID structure - now blittable
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct FormattedID
    {
        // Use fixed size arrays for string data (blittable)
        [System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] ParentID;  // Parent ID as bytes (ASCII)
        public int UniqueID;     // Unique non-reusable ID (already blittable)
        public byte ChildType;   // 'A' or 'B' as a byte (ASCII) instead of char

        public FormattedID(string parentID, int uniqueID, char childType)
        {
            ParentID = new byte[8]; // Fixed size array
            
            // Convert string to bytes (ASCII)
            if (!string.IsNullOrEmpty(parentID))
            {
                byte[] parentIDBytes = System.Text.Encoding.ASCII.GetBytes(parentID);
                int copyLength = Mathf.Min(parentIDBytes.Length, 8);
                System.Array.Copy(parentIDBytes, ParentID, copyLength);
            }
            
            UniqueID = uniqueID;
            ChildType = (byte)childType; // Convert char to byte
        }

        public override string ToString()
        {
            // Convert byte array to string
            string parentStr = System.Text.Encoding.ASCII.GetString(ParentID).TrimEnd('\0');
            return $"{parentStr}.{UniqueID:00}.{(char)ChildType}";
        }

        // Helper method to check if two cells have matching types
        public static bool HaveMatchingTypes(FormattedID id1, FormattedID id2)
        {
            return id1.ChildType == id2.ChildType;
        }
        
        // Helper method to get a readable string of the child type
        public static string GetTypeString(FormattedID id)
        {
            return ((char)id.ChildType).ToString();
        }

        // Parse a formatted ID string back into its components
        public static FormattedID Parse(string formattedID)
        {
            string[] parts = formattedID.Split('.');
            if (parts.Length < 3)
            {
                return new FormattedID("00", 0, 'A'); // Default if parsing fails
            }
            
            return new FormattedID(
                parts[0], 
                int.Parse(parts[1]), 
                parts[2][0]
            );
        }
    }

    public struct Cell
    {
        public Vector3 Position;
        public Vector3 Heading;
        public int Generation;
        public int ID;
        public FormattedID FormattedID; // Add formatted ID
        public bool MakeAdhesion;  // Parent's Make Adhesion flag
        public bool KeepAdhesion;  // Child's Keep Adhesion flag
        public Vector3 SplitPlaneNormal; // Normal of the parent's splitting plane in local space

        public Cell(Vector3 position, Vector3 heading, int generation, int id, FormattedID formattedID, bool makeAdhesion = true, bool keepAdhesion = true, Vector3 splitPlaneNormal = default(Vector3))
        {
            Position = position;
            Heading = heading.normalized;
            Generation = generation;
            ID = id;
            FormattedID = formattedID;
            MakeAdhesion = makeAdhesion;
            KeepAdhesion = keepAdhesion;
            SplitPlaneNormal = splitPlaneNormal == default(Vector3) ? Vector3.up : splitPlaneNormal.normalized;
        }
    }

    public struct Bond
    {
        public int A;
        public int B;

        public Bond(int a, int b)
        {
            A = Mathf.Min(a, b);
            B = Mathf.Max(a, b);
        }

        public override bool Equals(object obj)
        {
            if (!(obj is Bond))
                return false;
            Bond other = (Bond)obj;
            return A == other.A && B == other.B;
        }

        public override int GetHashCode()
        {
            return A.GetHashCode() ^ (B.GetHashCode() << 2);
        }
    }

    // New method to create adhesion connections based on particle data
    public static (Dictionary<int, Cell>, HashSet<Bond>) CreateAdhesionsFromParticles(
        Vector3[] particlePositions,
        Quaternion[] particleRotations,
        int activeParticleCount,
        ParticleIDData[] particleIDs,
        float maxConnectionDistance)
    {
        Debug.Log("Creating adhesion connections from particle data");

        // Don't run if we don't have particle positions yet
        if (particlePositions == null || particlePositions.Length == 0 || activeParticleCount <= 0)
        {
            Debug.LogWarning("Cannot create adhesion connections: no particle positions available.");
            return (new Dictionary<int, Cell>(), new HashSet<Bond>());
        }

        var cells = new Dictionary<int, Cell>();
        var bonds = new HashSet<Bond>();

        // Create cells for each active particle using their actual positions
        for (int i = 0; i < activeParticleCount; i++)
        {
            if (i >= particlePositions.Length) break;
            
            // Skip invalid positions
            if (float.IsNaN(particlePositions[i].x) || float.IsInfinity(particlePositions[i].x))
                continue;
            
            // Create a cell based on this particle's real position and ID
            Cell cell = new Cell
            {
                ID = i,
                Position = particlePositions[i],
                Heading = particleRotations[i] * Vector3.forward,
                Generation = 0  // All particles are treated as generation 0 for simplicity
            };

            // If particle IDs are available, set the formatted ID
            if (particleIDs != null && i < particleIDs.Length)
            {
                // Create formatted ID from particleIDs data
                cell.FormattedID = new FormattedID(
                    particleIDs[i].parentID.ToString("00"),
                    particleIDs[i].uniqueID,
                    particleIDs[i].childType
                );
            }
            
            cells[i] = cell;
        }
        
        Debug.Log($"Created {cells.Count} cells from particle positions.");
        
        // Create bonds based only on parent-child relationships, not proximity
        if (particleIDs != null)
        {
            for (int i = 0; i < activeParticleCount; i++)
            {
                if (!cells.ContainsKey(i)) continue;
                
                for (int j = i + 1; j < activeParticleCount; j++)
                {
                    if (!cells.ContainsKey(j)) continue;
                    
                    // Check if these particles are siblings (share the same parent)
                    if (particleIDs[i].parentID == particleIDs[j].parentID && 
                        particleIDs[i].parentID != 0) // Skip default value
                    {
                        bonds.Add(new Bond(i, j));
                    }
                }
            }
        }
        
        Debug.Log($"Final count: {cells.Count} cells and {bonds.Count} bonds.");
        return (cells, bonds);
    }
    
    // New method to visualize adhesion connections using LineRenderer
    public static void VisualizeConnections(
        LineRenderer lineRenderer,
        Dictionary<int, Cell> cells, 
        HashSet<Bond> bonds,
        Color bondColor)
    {
        Debug.Log("VisualizeConnections started");

        if (bonds.Count == 0 || lineRenderer == null)
        {
            Debug.Log("No bonds to visualize or missing LineRenderer");
            if (lineRenderer != null)
                lineRenderer.enabled = false;
            return;
        }

        // Configure the line renderer
        lineRenderer.positionCount = bonds.Count * 2;
        lineRenderer.startWidth = 0.02f;
        lineRenderer.endWidth = 0.02f;
        
        // Set color if material exists
        if (lineRenderer.material != null)
        {
            lineRenderer.material.color = bondColor;
        }
        else
        {
            // Create a default material if none exists
            lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
            lineRenderer.material.color = bondColor;
        }
        
        int index = 0;

        foreach (var bond in bonds)
        {
            Debug.Log($"Processing bond between {bond.A} and {bond.B}");

            if (cells.TryGetValue(bond.A, out var cellA) && cells.TryGetValue(bond.B, out var cellB))
            {
                Debug.Log($"Setting LineRenderer positions: {cellA.Position} -> {cellB.Position}");
                lineRenderer.SetPosition(index++, cellA.Position);
                lineRenderer.SetPosition(index++, cellB.Position);
            }
            else
            {
                Debug.LogWarning($"Failed to find cells for bond: {bond.A} <-> {bond.B}");
            }
        }

        lineRenderer.enabled = true;
        Debug.Log("VisualizeConnections completed");
    }
    
    // Structure for particle ID data - moved from ParticleSystemController
    public struct ParticleIDData
    {
        public int parentID;
        public int uniqueID;
        public char childType; // 'A' or 'B'

        public string GetFormattedID()
        {
            return $"{parentID:00}.{uniqueID:00}.{childType}";
        }
    }

    public static (Dictionary<int, Cell>, HashSet<Bond>) Simulate(
        int nGenerations, 
        Func<Cell, int, Vector3> childSplitPlaneNormalFn,  // Function to get split plane normal for each child
        Func<Cell, int, Vector3> childForwardDirectionFn,  // Function to get forward direction for each child
        Func<Cell, int, bool> childMakeAdhesionFn,         // Function to get MakeAdhesion flag for each child
        Func<Cell, int, bool> childKeepAdhesionFn          // Function to get KeepAdhesion flag for each child
    )
    {
        // Initialize with a single seed cell facing +X
        var cells = new Dictionary<int, Cell> { 
            { 0, new Cell(Vector3.zero, Vector3.right, 0, 0, new FormattedID("00", 0, 'A'), true, true, Vector3.up) } 
        };
        var bonds = new HashSet<Bond>();
        int nextID = 1;

        for (int gen = 1; gen <= nGenerations; gen++)
        {
            Debug.Log($"Processing generation {gen}");
            var oldBonds = new List<Bond>(bonds);
            bonds.Clear();
            var parents = new List<Cell>();

            // Collect parents of this generation
            foreach (var cell in cells.Values)
            {
                if (cell.Generation == gen - 1)
                {
                    parents.Add(cell);
                }
            }

            var spawnMap = new Dictionary<int, (Cell, Cell)>();
            var parentCellsToRemove = new List<int>();

            // 1) Split each parent into two children
            foreach (var parent in parents)
            {
                // Get the split plane normal in parent's local space
                Vector3 splitPlaneNormal = childSplitPlaneNormalFn(parent, 0);
                
                // Create orthogonal basis for splitting
                Vector3 parentHeading = parent.Heading;
                Vector3 splitDir = Vector3.Cross(parentHeading, splitPlaneNormal).normalized;

                // Child positions (could use configurable overlap)
                Vector3 posA = parent.Position + splitDir;
                Vector3 posB = parent.Position - splitDir;

                // Child forward directions in parent's local space
                Vector3 forwardA = childForwardDirectionFn(parent, 0);
                Vector3 forwardB = childForwardDirectionFn(parent, 1);

                // Make adhesion flags for each child
                bool makeAdhesionA = childMakeAdhesionFn(parent, 0);
                bool makeAdhesionB = childMakeAdhesionFn(parent, 1);

                // Keep adhesion flags for each child
                bool keepAdhesionA = childKeepAdhesionFn(parent, 0);
                bool keepAdhesionB = childKeepAdhesionFn(parent, 1);

                // Create child cells
                var childA = new Cell(
                    posA,
                    forwardA,
                    gen,
                    nextID++,
                    new FormattedID(parent.FormattedID.UniqueID.ToString("00"), nextID-1, 'A'), // Use parent's unique ID as origin
                    makeAdhesionA,
                    keepAdhesionA,
                    splitPlaneNormal
                );
                
                var childB = new Cell(
                    posB,
                    forwardB,
                    gen,
                    nextID++,
                    new FormattedID(parent.FormattedID.UniqueID.ToString("00"), nextID-1, 'B'), // Use parent's unique ID as origin
                    makeAdhesionB,
                    keepAdhesionB,
                    splitPlaneNormal
                );

                cells[childA.ID] = childA;
                cells[childB.ID] = childB;
                spawnMap[parent.ID] = (childA, childB);
                
                // 1️⃣ Sibling bond - create exactly one bond between A and B if parent's makeAdhesion is true
                if (parent.MakeAdhesion)
                {
                    Debug.Log($"Creating sibling bond between {childA.ID} and {childB.ID}");
                    bonds.Add(new Bond(childA.ID, childB.ID));
                }
                
                // Add parent to removal list instead of removing immediately
                parentCellsToRemove.Add(parent.ID);
            }
            
            // Process all newly created child cells for potential bonds with other new children
            foreach (var parentID1 in spawnMap.Keys)
            {
                var (childA1, childB1) = spawnMap[parentID1];
                
                foreach (var parentID2 in spawnMap.Keys)
                {
                    // Skip self-bonds
                    if (parentID1 == parentID2) continue;
                    
                    var (childA2, childB2) = spawnMap[parentID2];
                    
                    // Check for bonds between new children from different parents
                    TryCreateChildBond(childA1, childA2, bonds);
                    TryCreateChildBond(childA1, childB2, bonds);
                    TryCreateChildBond(childB1, childA2, bonds);
                    TryCreateChildBond(childB1, childB2, bonds);
                }
            }
            
            // 2️⃣ Inherit bonds from Parent→Neighbor
            foreach (var oldBond in oldBonds)
            {
                Debug.Log($"Processing old bond: {oldBond.A} <-> {oldBond.B}");
                int parentID = oldBond.A;
                int neighborID = oldBond.B;
                
                // Case 1: Both cells were parents that split
                if (spawnMap.TryGetValue(parentID, out var parentChildren) && 
                    spawnMap.TryGetValue(neighborID, out var neighborChildren))
                {
                    var (parentChildA, parentChildB) = parentChildren;
                    var (neighborChildA, neighborChildB) = neighborChildren;
                    
                    // Inherit bonds between corresponding same-type children, comparing types properly
                    if (parentChildA.KeepAdhesion && neighborChildA.KeepAdhesion && 
                        FormattedID.HaveMatchingTypes(parentChildA.FormattedID, neighborChildA.FormattedID))
                     {
                        Debug.Log($"Inheriting A-A bond between {parentChildA.ID}(type:{FormattedID.GetTypeString(parentChildA.FormattedID)}) and {neighborChildA.ID}(type:{FormattedID.GetTypeString(neighborChildA.FormattedID)})");
                         bonds.Add(new Bond(parentChildA.ID, neighborChildA.ID));
                     }
                    if (parentChildB.KeepAdhesion && neighborChildB.KeepAdhesion && 
                        FormattedID.HaveMatchingTypes(parentChildB.FormattedID, neighborChildB.FormattedID))
                     {
                        Debug.Log($"Inheriting B-B bond between {parentChildB.ID}(type:{FormattedID.GetTypeString(parentChildB.FormattedID)}) and {neighborChildB.ID}(type:{FormattedID.GetTypeString(neighborChildB.FormattedID)})");
                         bonds.Add(new Bond(parentChildB.ID, neighborChildB.ID));
                     }
                    continue;
                }
                
                // Case 2: First cell was a parent that split
                if (spawnMap.TryGetValue(parentID, out parentChildren))
                {
                    Debug.Log($"First end of bond (ID: {parentID}) was a parent that split, second end (ID: {neighborID}) remains");
                    var (childA, childB) = parentChildren;
                    
                    // Get the neighbor cell to check its type
                    if (cells.TryGetValue(neighborID, out Cell neighborCell))
                    {
                        char neighborType = (char)neighborCell.FormattedID.ChildType;
                        char childAType = (char)childA.FormattedID.ChildType;
                        char childBType = (char)childB.FormattedID.ChildType;
                        
                        // Test A: Check if child A should inherit the bond
                        if (childA.KeepAdhesion && childAType == neighborType) // Only keep bonds between same types
                        {
                            // Angle requirement commented out for testing
                            // float dotA = Vector3.Dot(childA.Heading, childA.SplitPlaneNormal);
                            Debug.Log($"Child A (ID: {childA.ID}, type: {childAType}) inherits bond to neighbor (ID: {neighborID}, type: {neighborType})");
                            bonds.Add(new Bond(childA.ID, neighborID));
                        }

                        // Test B: Check if child B should inherit the bond
                        if (childB.KeepAdhesion && childBType == neighborType)
                        {
                            // Angle requirement commented out for testing
                            // float dotB = Vector3.Dot(childB.Heading, childB.SplitPlaneNormal);
                            Debug.Log($"Child B (ID: {childB.ID}, type: {childBType}) inherits bond to neighbor (ID: {neighborID}, type: {neighborType})");
                            bonds.Add(new Bond(childB.ID, neighborID));
                        }
                    }
                    continue;
                }
                
                // Case 3: Second cell was a parent that split
                if (spawnMap.TryGetValue(neighborID, out var neighborChildrenPair))
                {
                    Debug.Log($"Second end of bond (ID: {neighborID}) was a parent that split, first end (ID: {parentID}) remains");
                    var (childA, childB) = neighborChildrenPair;
                    
                    // Get the parent cell to check its type
                    if (cells.TryGetValue(parentID, out Cell parentCell))
                    {
                        char parentType = (char)parentCell.FormattedID.ChildType;
                        char childAType = (char)childA.FormattedID.ChildType;
                        char childBType = (char)childB.FormattedID.ChildType;
                        
                        // Test A: Check if child A should inherit the bond
                        if (childA.KeepAdhesion && childAType == parentType) // Only keep bonds between same types
                        {
                            // Angle requirement commented out for testing
                            // float dotA = Vector3.Dot(childA.Heading, childA.SplitPlaneNormal);
                            Debug.Log($"Child A (ID: {childA.ID}, type: {childAType}) inherits bond to parent (ID: {parentID}, type: {parentType})");
                            bonds.Add(new Bond(childA.ID, parentID));
                        }
                        
                        // Test B: Check if child B should inherit the bond
                        if (childB.KeepAdhesion && childBType == parentType) // Only keep bonds between same types
                        {
                            // Angle requirement commented out for testing
                            // float dotB = Vector3.Dot(childB.Heading, childB.SplitPlaneNormal);
                            Debug.Log($"Child B (ID: {childB.ID}, type: {childBType}) inherits bond to parent (ID: {parentID}, type: {parentType})");
                            bonds.Add(new Bond(childB.ID, parentID));
                        }
                    }
                    continue;
                }
                
                // Case 4: Neither cell was a parent that split
                if (cells.TryGetValue(parentID, out var cellP) && cells.TryGetValue(neighborID, out var cellN))
                {
                    char typeP = (char)cellP.FormattedID.ChildType;
                    char typeN = (char)cellN.FormattedID.ChildType;
                    if (typeP == typeN)
                    {
                        Debug.Log($"Neither end was split: keeping bond between {parentID}(type:{typeP}) and {neighborID}(type:{typeN})");
                        bonds.Add(oldBond);
                    }
                }
                continue;
            }
            
            // Remove parents after processing all bonds
            foreach (var parentID in parentCellsToRemove)
            {
                cells.Remove(parentID);
            }
            
            Debug.Log($"Generation {gen} completed with {cells.Count} cells and {bonds.Count} bonds");
        }

        return (cells, bonds);
    }
    
    // Helper method to try creating a bond between two child cells
    private static void TryCreateChildBond(Cell childA, Cell childB, HashSet<Bond> bonds)
    {
        // Always allow formation of new links (MakeAdhesion check)
        if (childA.MakeAdhesion && childB.KeepAdhesion)
        {
            // Get cell types from FormattedIDs
            char typeA = (char)childA.FormattedID.ChildType;
            char typeB = (char)childB.FormattedID.ChildType;

            // Only form new bonds between same cell types (A-A or B-B)
            if (typeA != typeB)
                return;

            // Angle requirement commented out for testing
            // float dotA = Vector3.Dot(childA.Heading, childA.SplitPlaneNormal);
            // float dotB = Vector3.Dot(childB.Heading, childB.SplitPlaneNormal);

            Debug.Log($"Creating new child-to-child bond between {childA.ID}(type:{typeA}) and {childB.ID}(type:{typeB})");
            bonds.Add(new Bond(childA.ID, childB.ID));
        }
    }
}