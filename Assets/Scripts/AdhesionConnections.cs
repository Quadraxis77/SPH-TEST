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
            Sibling,  // Bond between cells from the same parent (newly created)
            Kept      // Bond that was inherited from a parent during cell division
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
            // Compare both indices AND bond type for true equality
            return A == other.A && B == other.B && Type == other.Type;
        }

        public override int GetHashCode()
        {
            // Include bond type in the hash code calculation
            return A.GetHashCode() ^ (B.GetHashCode() << 2) ^ ((int)Type << 4);
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
        // Ensure consistent calculations by rounding positions to reduce floating point variations
        Vector3 normalizedChildAPos = new Vector3(
            (float)Math.Round(childAPosition.x, 4),
            (float)Math.Round(childAPosition.y, 4),
            (float)Math.Round(childAPosition.z, 4)
        );
        
        Vector3 normalizedChildBPos = new Vector3(
            (float)Math.Round(childBPosition.x, 4),
            (float)Math.Round(childBPosition.y, 4),
            (float)Math.Round(childBPosition.z, 4)
        );
        
        Vector3 normalizedParentPos = new Vector3(
            (float)Math.Round(parentPosition.x, 4),
            (float)Math.Round(parentPosition.y, 4),
            (float)Math.Round(parentPosition.z, 4)
        );
        
        // Calculate the split direction using normalized positions
        Vector3 splitDirection = (normalizedChildBPos - normalizedChildAPos).normalized;
        
        // Round the split direction components to ensure consistency
        splitDirection = new Vector3(
            (float)Math.Round(splitDirection.x, 4),
            (float)Math.Round(splitDirection.y, 4),
            (float)Math.Round(splitDirection.z, 4)
        );
        
        // Normalize again after rounding to ensure unit length
        splitDirection = splitDirection.normalized;
        
        splitInfoMap[parentID] = new SplitInfo(parentID, splitDirection, normalizedParentPos);
    }

    /// <summary>
    /// Determines which child should inherit a bond based on a 30-degree trench aligned with the split plane
    /// </summary>
    private static bool ShouldInheritBond(int parentID, int childID, Vector3 childPos, Vector3 neighborPos, bool sharedBond)
    {
        // If we don't have split info for this parent, both children inherit bonds
        if (!splitInfoMap.TryGetValue(parentID, out SplitInfo splitInfo))
        {
            return true;
        }

        // Calculate vector from split position to neighbor
        Vector3 toNeighbor = neighborPos - splitInfo.splitPosition;
        
        // Create normalized bond direction vector
        Vector3 bondDir = toNeighbor.normalized;
        
        // Define the trench half angle (in degrees) - ensure consistency
        const float trenchHalfAngle = 9f; // Total trench angle will be 18 degrees (±9° from split plane)
        
        // Calculate the angle between the bond direction and the split plane
        // This is done by first finding the dot product with the split axis
        float dotWithSplitAxis = Vector3.Dot(bondDir, splitInfo.splitDirection);
        
        // The angle between the vector and the plane is the complement of the angle with the axis
        // Use exactly the same calculation everywhere for consistency
        float angleWithSplitPlane = 90f - Mathf.Acos(Mathf.Abs(dotWithSplitAxis)) * Mathf.Rad2Deg;
        
        // Round the angle to a small number of decimal places to avoid floating point inconsistencies
        angleWithSplitPlane = (float)Math.Round(angleWithSplitPlane, 4);
        
        // Determine which zone this bond belongs to
        string zone;
        
        // First check if it's in the trench zone
        if (angleWithSplitPlane <= trenchHalfAngle)
        {
            zone = "TRENCH ZONE";
            
            // For bonds in the trench zone, the side doesn't matter
            // Any child with keepAdhesion=true automatically inherits the bond
            // The keepAdhesion flag is already checked before calling this method
            return true;
        }
        
        // Calculate vector from split position to child
        Vector3 toChild = childPos - splitInfo.splitPosition;
        
        // For bonds outside the trench, assign to whichever child is on the same side 
        // Check if they're on the same side of the splitting plane
        float neighborSide = Vector3.Dot(toNeighbor, splitInfo.splitDirection);
        float childSide = Vector3.Dot(toChild, splitInfo.splitDirection);
        
        // Apply a small epsilon to avoid floating point comparison issues
        const float epsilon = 0.0001f;
        
        // Round both values to minimize floating point differences
        neighborSide = (float)Math.Round(neighborSide, 4);
        childSide = (float)Math.Round(childSide, 4);
        
        // More robust check for same side (both positive or both negative)
        bool sameSide = (childSide > epsilon && neighborSide > epsilon) || 
                       (childSide < -epsilon && neighborSide < -epsilon);
        
        // Determine which specific side this bond is on (A side or B side)
        if (neighborSide > epsilon)
            zone = "B SIDE"; // Positive dot product means B side (Child B direction)
        else if (neighborSide < -epsilon)
            zone = "A SIDE"; // Negative dot product means A side (opposite of Child B direction)
        else
            zone = "TRENCH ZONE"; // Very close to the split plane (should have been caught above, but just in case)
            
        // For non-trench zone bonds, maintain strict side assignment
        return sameSide;
    }

    /// <summary>
    /// Determines if a child should keep its parent's adhesion connections based on child type
    /// </summary>
    private static bool ShouldKeepAdhesion(char childType, int childIndex, ParticleSystemController controller)
    {
        if (controller != null && controller.genome != null)
        {
            // Get the specific mode for this particle
            GenomeMode genomeMode = controller.GetGenomeModeForParticle(childIndex);
            if (genomeMode != null)
            {
                if (childType == 'A')
                {
                    return genomeMode.childA_KeepAdhesion;
                }
                else if (childType == 'B')
                {
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
        ParticleSystemController controller,
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

        // 1. Create cells from all valid particles - process in fixed order
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
        
        // 2. Create valid connections based on sibling relationships
        if (particleIDs != null)
        {
            // Create a map of parent IDs to children for sibling connections
            var parentToChildren = new Dictionary<int, List<int>>();
            
            // Populate the parent-to-children map
            // First sort the cell IDs for consistent order of processing
            List<int> sortedCellIDs = new List<int>(cells.Keys);
            sortedCellIDs.Sort();
            
            foreach (int cellID in sortedCellIDs)
            {
                int parentID = particleIDs[cellID].parentID;
                if (!parentToChildren.ContainsKey(parentID))
                {
                    parentToChildren[parentID] = new List<int>();
                }
                parentToChildren[parentID].Add(cellID);
            }
            
            // RULE 1: Create bonds between siblings (same parent)
            // Sort parent IDs for consistent order of processing
            List<int> sortedParentIDs = new List<int>(parentToChildren.Keys);
            sortedParentIDs.Sort();
            
            foreach (int parentID in sortedParentIDs)
            {
                var children = parentToChildren[parentID];
                if (children.Count > 1)
                {
                    // Sort children to ensure consistent processing order
                    children.Sort();
                    
                    // Only connect direct siblings (typically A-B pairs)
                    for (int i = 0; i < children.Count; i++)
                    {
                        for (int j = i + 1; j < children.Count; j++)
                        {
                            int childA = children[i];
                            int childB = children[j];
                            
                            // Add sibling bond
                            bonds.Add(new Bond(childA, childB, Bond.BondType.Sibling));
                        }
                    }
                }
            }
            
            // Build the map of existing bonds for each parent
            foreach (int cellID in sortedCellIDs)
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
                // Create a map from uniqueIDs to current particle indices
                Dictionary<int, int> uniqueIdToIndex = new Dictionary<int, int>();
                for (int i = 0; i < activeParticleCount; i++)
                {
                    uniqueIdToIndex[particleIDs[i].uniqueID] = i;
                }
                
                // Process in consistent order by sorting the keys
                List<int> sortedUniqueIDs = new List<int>(_currentBondsByUniqueId.Keys);
                sortedUniqueIDs.Sort();
                
                // For each entry in the _currentBondsByUniqueId, map it to existing bonds
                foreach (int parentUniqueId in sortedUniqueIDs)
                {
                    // Check if we have a mapping from this uniqueId to an active cell
                    if (uniqueIdToIndex.TryGetValue(parentUniqueId, out int parentIndex))
                    {
                        int parentParentId = particleIDs[parentIndex].parentID;
                        
                        // Make sure the parent's entry exists in the dictionary
                        if (!existingBonds.ContainsKey(parentParentId))
                        {
                            existingBonds[parentParentId] = new List<int>();
                        }
                        
                        // Sort the connected IDs for consistent processing
                        List<int> sortedConnectedIDs = new List<int>(_currentBondsByUniqueId[parentUniqueId]);
                        sortedConnectedIDs.Sort();
                        
                        // Add connections
                        foreach (int connectedUniqueId in sortedConnectedIDs)
                        {
                            if (uniqueIdToIndex.TryGetValue(connectedUniqueId, out int connectedIndex))
                            {
                                // Only add if not already present
                                if (!existingBonds[parentParentId].Contains(connectedIndex))
                                {
                                    // Whether it was a sibling bond or not, add it to existing bonds to check inheritance
                                    existingBonds[parentParentId].Add(connectedIndex);
                                }
                            }
                        }
                    }
                }
            }
            
            // RULE 2: Inherit parent connections based on keepAdhesion flag
            // Sort parent IDs for processing in consistent order
            List<int> parentsToProcess = new List<int>(existingBonds.Keys);
            parentsToProcess.Sort();
            
            // We need to go through children within each parent and check their keepAdhesion flag
            foreach (int parentID in parentsToProcess)
            {
                if (parentToChildren.TryGetValue(parentID, out var children))
                {
                    // Sort children for consistent order of processing
                    children.Sort();
                    
                    foreach (int childID in children)
                    {
                        // Determine if this child should keep parent's adhesions
                        bool keepAdhesion = ShouldKeepAdhesion(particleIDs[childID].childType, childID, controller);
                        
                        if (keepAdhesion)
                        {
                            // Sort neighbors for consistent order of processing
                            List<int> sortedNeighbors = new List<int>(existingBonds[parentID]);
                            sortedNeighbors.Sort();
                            
                            // Inherit connections from parent
                            foreach (int neighborID in sortedNeighbors)
                            {
                                // Skip if it's already a sibling
                                if (parentToChildren.ContainsKey(parentID) && 
                                    parentToChildren[parentID].Contains(neighborID))
                                    continue;
                                    
                                // Skip self-connections
                                if (neighborID == childID)
                                    continue;
                                    
                                // Create test bonds for both possible types to check
                                Bond keptBond = new Bond(childID, neighborID, Bond.BondType.Kept);
                                Bond siblingBond = new Bond(childID, neighborID, Bond.BondType.Sibling);
                                
                                if (bonds.Contains(keptBond) || bonds.Contains(siblingBond))
                                {
                                    continue;
                                }
                                
                                // First determine if this child specifically has keepAdhesion enabled
                                bool thisChildKeepsAdhesion = ShouldKeepAdhesion(
                                    particleIDs[childID].childType, 
                                    childID, 
                                    controller
                                );
                                
                                // Skip if this child doesn't keep adhesion
                                if (!thisChildKeepsAdhesion)
                                {
                                    continue;
                                }
                                
                                // For the handshake, check if the neighbor also wants to keep connections
                                // Find the parent of the neighbor
                                int neighborParentID = particleIDs[neighborID].parentID;
                                bool neighborKeepsAdhesion = ShouldKeepAdhesion(
                                    particleIDs[neighborID].childType, 
                                    neighborID, 
                                    controller
                                );
                                
                                // Only proceed if both sides want to keep the connection
                                if (!neighborKeepsAdhesion)
                                {
                                    continue; // Skip if the other side doesn't want to keep the connection
                                }
                                
                                bool isSharedBond = false;
                                
                                // Check if all children of this parent have keepAdhesion enabled
                                if (parentToChildren.TryGetValue(parentID, out var siblingList) && siblingList.Count > 0)
                                {
                                    // Create a copy of the list before sorting to avoid modifying during enumeration
                                    List<int> siblingListCopy = new List<int>(siblingList);
                                    siblingListCopy.Sort(); // Sort for consistent processing
                                    
                                    bool allSiblingsKeepAdhesion = true;
                                    
                                    foreach (int siblingID in siblingListCopy)
                                    {
                                        bool siblingKeepsAdhesion = ShouldKeepAdhesion(
                                            particleIDs[siblingID].childType, 
                                            siblingID, 
                                            controller
                                        );
                                        
                                        if (!siblingKeepsAdhesion)
                                        {
                                            allSiblingsKeepAdhesion = false;
                                            break;
                                        }
                                    }
                                    
                                    isSharedBond = allSiblingsKeepAdhesion;
                                }
                                
                                // Check if this bond should be inherited based on position relative to split plane
                                bool shouldInherit = ShouldInheritBond(parentID, childID, cells[childID].Position, cells[neighborID].Position, isSharedBond);
                                
                                // If ShouldInheritBond returns false, it means the bond belongs to the other side
                                // Only add the bond if it should be inherited by this child
                                if (shouldInherit)
                                {
                                    bonds.Add(new Bond(childID, neighborID, Bond.BondType.Kept));
                                }
                            }
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

    // Dictionary to track bond types by unique ID pairs
    private static Dictionary<(int, int), Bond.BondType> _bondTypesByUniqueIdPair = new Dictionary<(int, int), Bond.BondType>();

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
        
        // Clear the dictionaries for this update
        _currentBondsByUniqueId.Clear();
        _bondTypesByUniqueIdPair.Clear();
        
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
            
            // Store bond type by the unique ID pair
            int minId = Mathf.Min(idA, idB);
            int maxId = Mathf.Max(idA, idB);
            _bondTypesByUniqueIdPair[(minId, maxId)] = bond.Type;
            
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

    /// <summary>
    /// Clears all static data to ensure clean state between simulation runs
    /// </summary>
    public static void ClearStaticData()
    {
        splitInfoMap.Clear();
        if (_currentBondsByUniqueId != null)
        {
            _currentBondsByUniqueId.Clear();
        }
        if (_bondTypesByUniqueIdPair != null)
        {
            _bondTypesByUniqueIdPair.Clear();
        }
        Debug.Log("Cleared all static adhesion data");
    }

    #endregion
}
