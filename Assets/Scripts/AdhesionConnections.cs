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
    /// Determines if a child should keep its parent's adhesion connections based on child type
    /// </summary>
    private static bool ShouldKeepAdhesion(char childType, int childIndex)
    {
        // Get the current genome and check the appropriate flag
        var controller = GameObject.FindObjectOfType<ParticleSystemController>();
        if (controller != null && controller.genome != null)
        {
            // Get the specific mode for this particle
            GenomeMode genomeMode = controller.GetGenomeModeForParticle(childIndex);
            if (genomeMode != null)
            {
                Debug.Log($"Checking keep adhesion for child type {childType}, mode: {genomeMode.modeName}");
                if (childType == 'A')
                {
                    Debug.Log($"Child A keepAdhesion = {genomeMode.childA_KeepAdhesion}");
                    return genomeMode.childA_KeepAdhesion;
                }
                else if (childType == 'B')
                {
                    Debug.Log($"Child B keepAdhesion = {genomeMode.childB_KeepAdhesion}");
                    return genomeMode.childB_KeepAdhesion;
                }
            }
            else
            {
                Debug.LogWarning($"No genome mode found for particle {childIndex}");
            }
        }
        
        // Default to true to ensure connections are maintained if we can't determine
        Debug.LogWarning("Could not determine whether to keep adhesion, defaulting to TRUE");
        return true;
    }

    /// <summary>
    /// Creates adhesion connections between particles based on their positions, rotations, and IDs
    /// </summary>
    public static (Dictionary<int, Cell>, HashSet<Bond>) CreateAdhesionsFromParticles(
        Vector3[] particlePositions,
        Quaternion[] particleRotations,
        int activeParticleCount,
        ParticleIDData[] particleIDs,
        float maxConnectionDistance) // Parameter kept for backward compatibility
    {
        if (particlePositions == null || particlePositions.Length == 0 || activeParticleCount <= 0)
        {
            Debug.LogWarning("Cannot create adhesion connections: no particle positions available.");
            return (new Dictionary<int, Cell>(), new HashSet<Bond>());
        }

        var cells = new Dictionary<int, Cell>();
        var bonds = new HashSet<Bond>();
        var existingBonds = new Dictionary<int, List<int>>(); // Track cell connections by parent

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
        
        // 2. Create valid connections based on sibling relationships
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
            
            // Build the map of existing bonds for each parent
            foreach (int cellID in cells.Keys)
            {
                int parentID = particleIDs[cellID].parentID;
                
                // Skip the root cell (parentID 0)
                if (parentID == 0) continue;
                
                if (!existingBonds.ContainsKey(parentID))
                {
                    existingBonds[parentID] = new List<int>();
                }
            }
            
            // Now look in the _currentBondsByUniqueId dictionary to find existing connections
            if (_currentBondsByUniqueId != null && _currentBondsByUniqueId.Count > 0)
            {
                Debug.Log($"Using {_currentBondsByUniqueId.Count} tracked bond connections");
                
                // Create a map from uniqueIDs to current particle indices
                Dictionary<int, int> uniqueIdToIndex = new Dictionary<int, int>();
                for (int i = 0; i < activeParticleCount; i++)
                {
                    uniqueIdToIndex[particleIDs[i].uniqueID] = i;
                }
                
                // For each entry in the _currentBondsByUniqueId, map it to existing bonds
                foreach (var entry in _currentBondsByUniqueId)
                {
                    int parentUniqueId = entry.Key;
                    
                    // Check if we have a mapping from this uniqueId to an active cell
                    if (uniqueIdToIndex.TryGetValue(parentUniqueId, out int parentIndex))
                    {
                        int parentParentId = particleIDs[parentIndex].parentID;
                        
                        // Make sure the parent's entry exists in the dictionary
                        if (!existingBonds.ContainsKey(parentParentId))
                        {
                            existingBonds[parentParentId] = new List<int>();
                        }
                        
                        // Add connections
                        foreach (int connectedUniqueId in entry.Value)
                        {
                            if (uniqueIdToIndex.TryGetValue(connectedUniqueId, out int connectedIndex))
                            {
                                // Only add if not already present
                                if (!existingBonds[parentParentId].Contains(connectedIndex))
                                {
                                    existingBonds[parentParentId].Add(connectedIndex);
                                    Debug.Log($"Added connection from parent {parentParentId} to {connectedIndex}");
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                Debug.Log("No existing bond connections found in tracking system");
            }
            
            // RULE 2: Inherit parent connections based on keepAdhesion flag
            // We need to go through children within each parent and check their keepAdhesion flag
            foreach (var parentID in existingBonds.Keys)
            {
                if (parentToChildren.TryGetValue(parentID, out var children))
                {
                    foreach (int childID in children)
                    {
                        // Determine if this child should keep parent's adhesions
                        bool keepAdhesion = ShouldKeepAdhesion(particleIDs[childID].childType, childID);
                        Debug.Log($"Child {particleIDs[childID].GetFormattedID()} keep adhesion: {keepAdhesion}");
                        
                        if (keepAdhesion)
                        {
                            // Inherit connections from parent
                            foreach (int neighborID in existingBonds[parentID])
                            {
                                // Skip if it's already a sibling
                                if (parentToChildren.ContainsKey(parentID) && 
                                    parentToChildren[parentID].Contains(neighborID))
                                    continue;
                                    
                                // Skip self-connections
                                if (neighborID == childID)
                                    continue;
                                    
                                // Skip if we already have this bond
                                Bond newBond = new Bond(childID, neighborID, Bond.BondType.SameType);
                                if (bonds.Contains(newBond))
                                    continue;
                                    
                                // Check if child and neighbor are on the same side of the split
                                if (ShouldInheritBond(parentID, childID, cells[childID].Position, cells[neighborID].Position))
                                {
                                    bonds.Add(newBond);
                                    Debug.Log($"Inherited parent bond: {particleIDs[childID].GetFormattedID()} to {particleIDs[neighborID].GetFormattedID()}");
                                }
                            }
                        }
                        else
                        {
                            Debug.Log($"Child {particleIDs[childID].GetFormattedID()} NOT keeping adhesion connections from parent");
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

    #region Bond Tracking

    /// <summary>
    /// Function to expose the current bonds to the controller
    /// </summary>
    public static void SetCurrentBonds(ParticleSystemController controller, HashSet<Bond> currentBonds)
    {
        // Create a static dictionary to track bonds between cells by their uniqueIDs rather than indices
        if (_currentBondsByUniqueId == null)
        {
            _currentBondsByUniqueId = new Dictionary<int, List<int>>();
        }
        
        // Clear the dictionary for this update
        _currentBondsByUniqueId.Clear();
        
        if (currentBonds == null || currentBonds.Count == 0)
        {
            return;
        }
        
        // Get the particle IDs from the controller
        var particleIDs = controller.GetParticleIDs();
        if (particleIDs == null)
        {
            Debug.LogWarning("Cannot set current bonds: no particle IDs available");
            return;
        }
        
        // Fill the dictionary with all current bonds using uniqueIDs
        foreach (var bond in currentBonds)
        {
            int idA = particleIDs[bond.A].uniqueID;
            int idB = particleIDs[bond.B].uniqueID;
            
            // Add A's connection to B
            if (!_currentBondsByUniqueId.ContainsKey(idA))
            {
                _currentBondsByUniqueId[idA] = new List<int>();
            }
            _currentBondsByUniqueId[idA].Add(idB);
            
            // Add B's connection to A
            if (!_currentBondsByUniqueId.ContainsKey(idB))
            {
                _currentBondsByUniqueId[idB] = new List<int>();
            }
            _currentBondsByUniqueId[idB].Add(idA);
        }
        
        Debug.Log($"Set {_currentBondsByUniqueId.Count} cells with connections in the bond tracking system");
    }
    
    // Static dictionary to track bonds using uniqueIDs instead of indices
    private static Dictionary<int, List<int>> _currentBondsByUniqueId;

    #endregion
}
