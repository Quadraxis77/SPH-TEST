using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Handles the creation and management of adhesion connections between cells in a particle system.
/// </summary>
public class AdhesionConnections
{
    #region Data Structures

    /// <summary>
    /// Structured ID format for particles that allows for parent-child relationship tracking
    /// </summary>
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct FormattedID
    {
        // Fixed size array for blittable data structure
        [System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] ParentID;  // Parent ID as bytes (ASCII)
        public int UniqueID;     // Unique non-reusable ID
        public byte ChildType;   // 'A' or 'B' as a byte (ASCII)

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
            ChildType = (byte)childType;
        }

        public override string ToString()
        {
            string parentStr = System.Text.Encoding.ASCII.GetString(ParentID).TrimEnd('\0');
            return $"{parentStr}.{UniqueID:00}.{(char)ChildType}";
        }

        /// <summary>
        /// Returns true if the two IDs have the same child type (both A or both B)
        /// </summary>
        public static bool HaveMatchingTypes(FormattedID id1, FormattedID id2)
        {
            return id1.ChildType == id2.ChildType;
        }
        
        /// <summary>
        /// Returns the child type as a string
        /// </summary>
        public static string GetTypeString(FormattedID id)
        {
            return ((char)id.ChildType).ToString();
        }

        /// <summary>
        /// Parse a formatted ID string into a FormattedID struct
        /// </summary>
        public static FormattedID Parse(string formattedID)
        {
            string[] parts = formattedID.Split('.');
            if (parts.Length < 3)
            {
                return new FormattedID("00", 0, 'A'); // Default
            }
            
            return new FormattedID(
                parts[0], 
                int.Parse(parts[1]), 
                parts[2][0]
            );
        }
    }

    /// <summary>
    /// Represents a cell in the simulation
    /// </summary>
    public struct Cell
    {
        public Vector3 Position;
        public Vector3 Heading;
        public int Generation;
        public int ID;
        public FormattedID FormattedID;
        public bool CanFormAdhesion;  // Whether this cell can form new adhesion bonds
        public char Type => (char)FormattedID.ChildType;  // Convenience property to get child type

        public Cell(Vector3 position, Vector3 heading, int generation, int id, FormattedID formattedID, bool canFormAdhesion = true)
        {
            Position = position;
            Heading = heading.normalized;
            Generation = generation;
            ID = id;
            FormattedID = formattedID;
            CanFormAdhesion = canFormAdhesion;
        }
    }

    /// <summary>
    /// Represents an adhesion bond between two cells
    /// </summary>
    public struct Bond
    {
        public int A;
        public int B;
        public BondType Type;

        public enum BondType
        {
            Sibling,  // Bond between cells from the same parent
            SameType  // Bond between same-type cells (A-A or B-B)
        }

        public Bond(int a, int b, BondType type)
        {
            // Always store indices in ascending order for consistent hash codes
            A = Mathf.Min(a, b);
            B = Mathf.Max(a, b);
            Type = type;
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

    /// <summary>
    /// Data structure for particle ID data
    /// </summary>
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

    // Data structure to store splitting information
    public struct SplitInfo
    {
        public int parentID;
        public Vector3 splitDirection;  // Normalized direction from child A to child B
        public Vector3 splitPosition;   // Position where the split occurred (parent position)

        public SplitInfo(int parentID, Vector3 splitDirection, Vector3 splitPosition)
        {
            this.parentID = parentID;
            this.splitDirection = splitDirection.normalized;
            this.splitPosition = splitPosition;
        }
    }

    // Dictionary to track split information for each parent
    private static Dictionary<int, SplitInfo> splitInfoMap = new Dictionary<int, SplitInfo>();

    #endregion

    #region Connection Creation

    /// <summary>
    /// Registers a cell division event to track split direction
    /// </summary>
    public static void RegisterCellDivision(int parentID, Vector3 parentPosition, Vector3 childAPosition, Vector3 childBPosition)
    {
        Vector3 splitDirection = (childBPosition - childAPosition).normalized;
        splitInfoMap[parentID] = new SplitInfo(parentID, splitDirection, parentPosition);
    }

    /// <summary>
    /// Determines which child should inherit a bond based on half-side check
    /// </summary>
    private static bool ShouldInheritBond(int parentID, int childID, Vector3 childPos, Vector3 neighborPos)
    {
        // If we don't have split info for this parent, both children inherit bonds
        if (!splitInfoMap.TryGetValue(parentID, out SplitInfo splitInfo))
        {
            return true;
        }

        // Calculate vector from split position to neighbor
        Vector3 toNeighbor = neighborPos - splitInfo.splitPosition;
        
        // Calculate vector from split position to child
        Vector3 toChild = childPos - splitInfo.splitPosition;
        
        // Check if they're on the same side of the splitting plane
        // Dot product will be positive if they're on the same side
        float neighborSide = Vector3.Dot(toNeighbor, splitInfo.splitDirection);
        float childSide = Vector3.Dot(toChild, splitInfo.splitDirection);
        
        // Child inherits bond if it's on the same side as the neighbor
        return (neighborSide * childSide > 0);
    }

    /// <summary>
    /// Creates adhesion connections between particles based on their positions, rotations, and IDs
    /// </summary>
    public static (Dictionary<int, Cell>, HashSet<Bond>) CreateAdhesionsFromParticles(
        Vector3[] particlePositions,
        Quaternion[] particleRotations,
        int activeParticleCount,
        ParticleIDData[] particleIDs,
        float maxConnectionDistance)
    {
        if (particlePositions == null || particlePositions.Length == 0 || activeParticleCount <= 0)
        {
            Debug.LogWarning("Cannot create adhesion connections: no particle positions available.");
            return (new Dictionary<int, Cell>(), new HashSet<Bond>());
        }

        var cells = new Dictionary<int, Cell>();
        var bonds = new HashSet<Bond>();

        // 1. Create cells from all valid particles
        for (int i = 0; i < activeParticleCount; i++)
        {
            if (i >= particlePositions.Length) break;
            
            // Skip invalid positions
            if (float.IsNaN(particlePositions[i].x) || float.IsInfinity(particlePositions[i].x))
                continue;
            
            Cell cell = new Cell
            {
                ID = i,
                Position = particlePositions[i],
                Heading = particleRotations[i] * Vector3.forward,
                Generation = 0,
                CanFormAdhesion = true
            };

            // Set formatted ID if available
            if (particleIDs != null && i < particleIDs.Length)
            {
                cell.FormattedID = new FormattedID(
                    particleIDs[i].parentID.ToString("00"),
                    particleIDs[i].uniqueID,
                    particleIDs[i].childType
                );
            }
            
            cells[i] = cell;
        }
        
        Debug.Log($"Created {cells.Count} cells from particle positions.");
        
        // 2. Create only strictly valid connections
        if (particleIDs != null)
        {
            // Create a map of parent IDs to children for sibling connections
            var parentToChildren = new Dictionary<int, List<int>>();
            
            // Populate the parent-to-children map
            foreach (int cellID in cells.Keys)
            {
                int parentID = particleIDs[cellID].parentID;
                if (!parentToChildren.ContainsKey(parentID))
                {
                    parentToChildren[parentID] = new List<int>();
                }
                parentToChildren[parentID].Add(cellID);
            }
            
            // RULE 1: Create bonds between siblings (same parent)
            foreach (var children in parentToChildren.Values)
            {
                if (children.Count > 1)
                {
                    // Only connect direct siblings (typically A-B pairs)
                    for (int i = 0; i < children.Count; i++)
                    {
                        for (int j = i + 1; j < children.Count; j++)
                        {
                            int childA = children[i];
                            int childB = children[j];
                            
                            // Add sibling bond
                            bonds.Add(new Bond(childA, childB, Bond.BondType.Sibling));
                            Debug.Log($"Created SIBLING bond between {particleIDs[childA].GetFormattedID()} and {particleIDs[childB].GetFormattedID()}");
                        }
                    }
                }
            }
            
            // RULE 2: Create bonds between same-type cells (A-A or B-B)
            // Use a spatial grid for efficient neighbor finding
            var cellGrid = CreateSpatialGrid(cells, maxConnectionDistance);
            
            foreach (int cellID in cells.Keys)
            {
                char cellType = particleIDs[cellID].childType;
                Vector3 cellPos = cells[cellID].Position;
                
                // Find potential neighbors using the spatial grid
                var potentialNeighbors = GetPotentialNeighbors(cellPos, cellGrid, maxConnectionDistance);
                
                foreach (int neighborID in potentialNeighbors)
                {
                    // Skip self and already processed pairs
                    if (neighborID <= cellID) continue;
                    
                    // Skip if these are already siblings
                    if (particleIDs[cellID].parentID == particleIDs[neighborID].parentID)
                        continue;
                    
                    // Only connect same types (A-A or B-B)
                    if (particleIDs[neighborID].childType == cellType)
                    {
                        // Check distance
                        float distance = Vector3.Distance(cellPos, cells[neighborID].Position);
                        if (distance <= maxConnectionDistance)
                        {
                            bonds.Add(new Bond(cellID, neighborID, Bond.BondType.SameType));
                            Debug.Log($"Created SAME_TYPE bond between {particleIDs[cellID].GetFormattedID()} and {particleIDs[neighborID].GetFormattedID()}");
                        }
                    }
                }
            }
        }
        
        Debug.Log($"Final results: {cells.Count} cells with {bonds.Count} adhesion bonds");
        return (cells, bonds);
    }

    /// <summary>
    /// Creates a spatial grid for more efficient neighbor finding
    /// </summary>
    private static Dictionary<Vector3Int, List<int>> CreateSpatialGrid(Dictionary<int, Cell> cells, float cellSize)
    {
        var grid = new Dictionary<Vector3Int, List<int>>();
        
        foreach (var kvp in cells)
        {
            int cellID = kvp.Key;
            Vector3 position = kvp.Value.Position;
            
            // Get grid cell coordinate
            Vector3Int gridCoord = new Vector3Int(
                Mathf.FloorToInt(position.x / cellSize),
                Mathf.FloorToInt(position.y / cellSize),
                Mathf.FloorToInt(position.z / cellSize)
            );
            
            // Add to grid
            if (!grid.ContainsKey(gridCoord))
            {
                grid[gridCoord] = new List<int>();
            }
            grid[gridCoord].Add(cellID);
        }
        
        return grid;
    }

    /// <summary>
    /// Gets potential neighbors from the spatial grid
    /// </summary>
    private static HashSet<int> GetPotentialNeighbors(Vector3 position, Dictionary<Vector3Int, List<int>> grid, float cellSize)
    {
        var neighbors = new HashSet<int>();
        Vector3Int gridPos = new Vector3Int(
            Mathf.FloorToInt(position.x / cellSize),
            Mathf.FloorToInt(position.y / cellSize),
            Mathf.FloorToInt(position.z / cellSize)
        );
        
        // Check current grid cell and adjacent cells (27 cells in 3D)
        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int z = -1; z <= 1; z++)
                {
                    Vector3Int checkPos = gridPos + new Vector3Int(x, y, z);
                    if (grid.TryGetValue(checkPos, out List<int> cellsInGrid))
                    {
                        foreach (int cellID in cellsInGrid)
                        {
                            neighbors.Add(cellID);
                        }
                    }
                }
            }
        }
        
        return neighbors;
    }

    #endregion

    #region Visualization

    /// <summary>
    /// Visualizes adhesion connections using LineRenderer
    /// </summary>
    public static void VisualizeConnections(
        LineRenderer lineRenderer,
        Dictionary<int, Cell> cells, 
        HashSet<Bond> bonds,
        Color bondColor)
    {
        if (bonds.Count == 0 || lineRenderer == null)
        {
            if (lineRenderer != null)
                lineRenderer.enabled = false;
            return;
        }

        // Filter out invalid bonds
        List<(Vector3, Vector3, Bond.BondType)> validBonds = new List<(Vector3, Vector3, Bond.BondType)>();
        
        foreach (var bond in bonds)
        {
            if (cells.TryGetValue(bond.A, out var cellA) && cells.TryGetValue(bond.B, out var cellB))
            {
                // Only include bonds with valid positions
                if (IsValidPosition(cellA.Position) && IsValidPosition(cellB.Position))
                {
                    validBonds.Add((cellA.Position, cellB.Position, bond.Type));
                }
            }
        }
        
        // If no valid bonds, disable the renderer and return
        if (validBonds.Count == 0)
        {
            lineRenderer.enabled = false;
            return;
        }
        
        // Configure the line renderer
        lineRenderer.positionCount = validBonds.Count * 2;
        lineRenderer.startWidth = 0.05f;
        lineRenderer.endWidth = 0.05f;
        
        // Set color
        if (lineRenderer.material == null)
        {
            lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        }
        lineRenderer.material.color = bondColor;
        
        // Set positions
        int index = 0;
        foreach (var (posA, posB, _) in validBonds)
        {
            lineRenderer.SetPosition(index++, posA);
            lineRenderer.SetPosition(index++, posB);
        }

        lineRenderer.enabled = true;
    }

    /// <summary>
    /// Checks if a Vector3 position is valid
    /// </summary>
    private static bool IsValidPosition(Vector3 position)
    {
        return !float.IsNaN(position.x) && !float.IsNaN(position.y) && !float.IsNaN(position.z) &&
               !float.IsInfinity(position.x) && !float.IsInfinity(position.y) && !float.IsInfinity(position.z);
    }

    #endregion
}