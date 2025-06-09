using UnityEngine;
using System.Collections.Generic;
using UnityEngine.Rendering;  // Add this for AsyncGPUReadback
using System.Linq;
using TMPro;

public class ParticleSystemController : MonoBehaviour
{
    #region Public Properties

    [Header("Particle Configuration")]
    public int particleCount = 10000;
    public float minRadius = 1.5f;
    public float maxRadius = 2.0f;
    public float spawnRadius = 15f;

    [Header("Simulation Settings")]
    [Range(0f, 10f)] public float globalDragMultiplier = 1.0f;
    [Range(0f, 50f)] public float torqueFactor = 1.0f;
    [Range(0f, 10f)] public float torqueDamping = 0.5f;
    [Range(0f, 10f)] public float boundaryFriction = 0.8f;
    [Range(1f, 10f)] public float rollingContactRadiusMultiplier = 5.0f;
    [Range(0f, 10f)] public float density = 0.1f;
    [Range(0f, 500f)] public float repulsionStrength = 200.0f;

    [Header("Cell Division Settings")]
    [Range(0.1f, 5f)] public float spawnOverlapOffset = 0.5f;
    [Range(0.1f, 10f)] public float splitVelocityMagnitude = 0.5f;
    // Removed globalSplitTimerMin - now using mode-specific split intervals

    [Header("Genome Settings")]
    public CellGenome genome;

    [Header("Simulation Assets")]
    public ComputeShader computeShader;
    public Material sphereMaterial;
    public Mesh sphereMesh;

    [Header("Drag Visualization Settings")]
    public Color dragCircleColor = Color.green;
    public float dragCircleRadius = 1.0f;

    [Header("Text Labels")]
    public bool showParticleLabels = true;
    public float labelOffset = 2.5f;
    public float labelScale = 0.5f;
    public Color labelColor = Color.white;
    
    [Header("Split Plane Ring Visualization")]
    public bool showSplitPlaneRings = false;
    public float splitPlaneRingRadius = 2.0f;
    public int splitPlaneRingSegments = 48;
    public Color splitPlaneRingColor = Color.cyan;
    
    [Header("Adhesion Visualization")]
    public CellAdhesionManager adhesionManager;
    
    #endregion

    #region Private Fields
    
    private LineRenderer circleRenderer;
    private LineRenderer lineRenderer;

    ComputeBuffer particleBuffer, dragInputBuffer, drawArgsBufferSpheres;
    ComputeBuffer positionReadbackBuffer, rotationReadbackBuffer;
    ComputeBuffer gridHeads, gridNext, gridParticleIndices;
    ComputeBuffer torqueAccumBuffer;
    ComputeBuffer adhesionConnectionBuffer;
    ComputeBuffer adhesionVelocityDeltaBuffer;
    ComputeBuffer adhesionRotationDeltaBuffer;

    const int GRID_DIM = 32;
    const int GRID_TOTAL = GRID_DIM * GRID_DIM * GRID_DIM;

    int kernelInitParticles;
    int kernelClearGrid;
    int kernelBuildGrid;
    int kernelApplySPHForces;
    int kernelApplyDrag;
    int kernelUpdateMotion;
    int kernelUpdateRotation;
    int kernelCopyPositions;    int kernelCopyRotations;
    int kernelApplyAdhesionConstraints;
    int kernelApplyAdhesionDeltas;

    int selectedParticleID = -1;
    Vector3 dragTargetWorld;
    private Vector3[] cpuParticlePositions;
    public Vector3[] CpuParticlePositions => cpuParticlePositions;
    private Quaternion[] cpuParticleRotations;
    public Quaternion[] CpuParticleRotations => cpuParticleRotations;

    private float currentDragDistance;
    private int activeParticleCount = 1; // Class-level variable to track active particles

    // Global counter to ensure unique IDs are never reused
    private int nextUniqueIDCounter = 1; // Start at 1 since first particle (0) is created in Start
    
    // Timer variables for cell splitting
    private float[] cellSplitTimers;
    
    // List to track pending cell splits
    private List<CellSplitData> pendingSplits = new List<CellSplitData>();

    // Add member variables to track readback requests
    private bool particleDataReadbackInProgress = false;
    private Particle[] cachedParticleData;
    
    // Add a flag to track if our cached particle data is ready/valid
    private bool cachedParticleDataValid = false;

    // Particle label array
    private TextMeshPro[] particleLabels;    // Add a new buffer to hold genome mode data for the shader
    private ComputeBuffer genomeModesBuffer;
    
    // Array to store formatted IDs for each particle
    private ParticleIDData[] particleIDs;
    public ParticleIDData[] ParticleIDs => particleIDs;

    // Array to hold split plane ring renderers
    private LineRenderer[] splitPlaneRings;
    
    // Add a field to track the most recently dragged/selected cell
    private int lastSelectedParticleID = -1;
    public int LastSelectedParticleID => lastSelectedParticleID;
    
    // Maximum number of adhesion connections (arbitrary limit for now)
    int maxAdhesionConnections = 4096;
    
    #endregion

    #region Data Structures
    
    // Struct for cell division data
    private struct CellSplitData
    {
        public int parentIndex;
        public Vector3 positionA;
        public Vector3 positionB;
        public Vector3 velocityA;
        public Vector3 velocityB;
        public Quaternion rotationA;
        public Quaternion rotationB;
        public int childAModeIndex;
        public int childBModeIndex;
    }

    struct DragInput
    {
        public int selectedID;
        public Vector3 targetPosition;
        public float strength;
    }
    
    // The Particle struct used to match with the compute shader definition
    struct Particle
    {
        public Vector3 position;
        public float radius;
        
        public Vector3 velocity;
        public float mass;
        
        public Vector3 angularVelocity;
        public float momentOfInertia;
        
        public float drag;
        public float repulsionStrength;
        public uint genomeFlags;
        public float orientConstraintStr;
        
        public Quaternion rotation;
        public int modeIndex; // Added field to store the mode index
    }
    
    // Custom struct to store particle ID data
    public struct ParticleIDData
    {
        public int parentID;
        public int uniqueID;
        public char childType;
        
        public string GetFormattedID()
        {
            if (childType == '\0') // Default value for char
                return "Unknown";
                
            return $"{parentID:D2}.{uniqueID:D2}.{childType}";
        }
    }
      
    // Struct that matches the GenomeAdhesionData in the shader (fields up to colorPacked)
    private struct GenomeColorData
    {
        public int parentMakeAdhesion;
        public int childA_KeepAdhesion;
        public int childB_KeepAdhesion;
        public float adhesionRestLength;
        public float adhesionSpringStiffness;
        public float adhesionSpringDamping;
        public uint colorPacked;
        public float orientConstraintStrength;
        public float maxAngleDeviation;
    }
    
    #endregion

    #region Unity Lifecycle Methods
    
    void Start()
    {
        Application.targetFrameRate = 144;
        
        // Subscribe to genome changes
        if (genome != null)
        {
            // Validate the genome configuration before starting the simulation
            try
            {
                genome.ValidateForSimulation();
            }            catch (System.InvalidOperationException)
            {
                enabled = false; // Disable this component to prevent the simulation from running
                return;
            }
            
            CellGenome.OnGenomeChanged += OnGenomeChanged;
        }

        // Initialize all buffers
        InitializeBuffers();
        
        // Create and initialize the genome modes buffer for colors
        UpdateGenomeModesBuffer();
          // Initialize particles with genome properties
        InitializeParticles();
        
        // Initialize particle labels
        InitializeParticleLabels();        // Initialize split plane rings
        InitializeSplitPlaneRings();
    }

    void Update()
    {
        float dt = Time.deltaTime;
        int threadGroups = Mathf.CeilToInt(particleCount / 64f);

        // Request asynchronous readback of particle data if needed
        RequestParticleDataAsync();

        // Update cell split timers and handle cell division
        UpdateCellDivisionTimers(dt);

        computeShader.SetFloat("deltaTime", dt);
        computeShader.SetFloat("globalDragMultiplier", globalDragMultiplier);
        computeShader.SetFloat("torqueFactor", torqueFactor);
        computeShader.SetFloat("torqueDamping", torqueDamping);
        computeShader.SetFloat("boundaryFriction", boundaryFriction);
        computeShader.SetFloat("rollingContactRadiusMultiplier", rollingContactRadiusMultiplier);
        computeShader.SetFloat("density", density);
        computeShader.SetFloat("repulsionStrength", repulsionStrength);
        computeShader.SetInt("activeParticleCount", activeParticleCount); // Make sure to set this every frame
        
        torqueAccumBuffer.SetData(new int[particleCount * 3]);

        uint[] clear = new uint[GRID_TOTAL];
        for (int i = 0; i < GRID_TOTAL; i++) clear[i] = 0xffffffff;
        gridHeads.SetData(clear);
        computeShader.SetBuffer(kernelClearGrid, "gridHeads", gridHeads);
        computeShader.Dispatch(kernelClearGrid, Mathf.CeilToInt(GRID_TOTAL / 64f), 1, 1);

        computeShader.SetBuffer(kernelBuildGrid, "particleBuffer", particleBuffer);
        computeShader.SetBuffer(kernelBuildGrid, "gridHeads", gridHeads);
        computeShader.SetBuffer(kernelBuildGrid, "gridNext", gridNext);
        computeShader.SetBuffer(kernelBuildGrid, "gridParticleIndices", gridParticleIndices);
        computeShader.Dispatch(kernelBuildGrid, threadGroups, 1, 1);

        computeShader.SetBuffer(kernelApplySPHForces, "particleBuffer", particleBuffer);
        computeShader.SetBuffer(kernelApplySPHForces, "gridHeads", gridHeads);
        computeShader.SetBuffer(kernelApplySPHForces, "gridNext", gridNext);
        computeShader.SetBuffer(kernelApplySPHForces, "gridParticleIndices", gridParticleIndices);
        computeShader.SetBuffer(kernelApplySPHForces, "torqueAccumBuffer", torqueAccumBuffer);
        computeShader.Dispatch(kernelApplySPHForces, threadGroups, 1, 1);        // --- Apply adhesion constraints immediately after SPH forces ---
        if (adhesionManager != null && adhesionConnectionBuffer != null){
            var connections = adhesionManager.GetAdhesionConnectionsForGPU();
            int count = Mathf.Min(connections.Length, maxAdhesionConnections);
            
            if (count > 0)
            {
                adhesionConnectionBuffer.SetData(connections, 0, 0, count);
                // Clear delta buffers
                adhesionVelocityDeltaBuffer.SetData(new int[particleCount * 3]);
                adhesionRotationDeltaBuffer.SetData(new int[particleCount * 4]);

                computeShader.SetBuffer(kernelApplyAdhesionConstraints, "adhesionConnectionBuffer", adhesionConnectionBuffer);
                computeShader.SetInt("adhesionConnectionCount", count);
                computeShader.SetBuffer(kernelApplyAdhesionConstraints, "particleBuffer", particleBuffer);
                computeShader.SetBuffer(kernelApplyAdhesionConstraints, "adhesionVelocityDeltaBuffer", adhesionVelocityDeltaBuffer);
                computeShader.SetBuffer(kernelApplyAdhesionConstraints, "adhesionRotationDeltaBuffer", adhesionRotationDeltaBuffer);
                // Explicitly set deltaTime for the adhesion constraints kernel
                computeShader.SetFloat("deltaTime", dt);
                computeShader.Dispatch(kernelApplyAdhesionConstraints, count, 1, 1);
                  // Apply deltas to each particle
                computeShader.SetBuffer(kernelApplyAdhesionDeltas, "particleBuffer", particleBuffer);
                computeShader.SetBuffer(kernelApplyAdhesionDeltas, "adhesionVelocityDeltaBuffer", adhesionVelocityDeltaBuffer);
                computeShader.SetBuffer(kernelApplyAdhesionDeltas, "adhesionRotationDeltaBuffer", adhesionRotationDeltaBuffer);                computeShader.SetFloat("deltaTime", dt);
                computeShader.Dispatch(kernelApplyAdhesionDeltas, threadGroups, 1, 1);
            }
        }
        // --- End adhesion constraints ---

        computeShader.SetBuffer(kernelApplyDrag, "particleBuffer", particleBuffer);
        computeShader.SetBuffer(kernelApplyDrag, "dragInput", dragInputBuffer);
        HandleMouseDrag();
        computeShader.Dispatch(kernelApplyDrag, 1, 1, 1);

        computeShader.SetBuffer(kernelUpdateMotion, "particleBuffer", particleBuffer);
        computeShader.Dispatch(kernelUpdateMotion, threadGroups, 1, 1);

        computeShader.SetBuffer(kernelUpdateRotation, "particleBuffer", particleBuffer);
        computeShader.SetBuffer(kernelUpdateRotation, "torqueAccumBuffer", torqueAccumBuffer);
        computeShader.Dispatch(kernelUpdateRotation, threadGroups, 1, 1);

        computeShader.SetBuffer(kernelCopyPositions, "particleBuffer", particleBuffer);
        computeShader.SetBuffer(kernelCopyPositions, "positionReadbackBuffer", positionReadbackBuffer);
        computeShader.Dispatch(kernelCopyPositions, threadGroups, 1, 1);

        computeShader.SetBuffer(kernelCopyRotations, "particleBuffer", particleBuffer);
        computeShader.SetBuffer(kernelCopyRotations, "rotationReadbackBuffer", rotationReadbackBuffer);
        computeShader.Dispatch(kernelCopyRotations, threadGroups, 1, 1);        // Immediate readback of position and rotation data for rendering
        positionReadbackBuffer.GetData(cpuParticlePositions);
        rotationReadbackBuffer.GetData(cpuParticleRotations);

        uint[] args = new uint[5];
        drawArgsBufferSpheres.GetData(args);
        args[1] = (uint)activeParticleCount; // Use activeParticleCount instead of total count
        drawArgsBufferSpheres.SetData(args);

        sphereMaterial.SetBuffer("particleBuffer", particleBuffer);
        // Ensure GPU instancing is enabled on the material for per-instance properties
        sphereMaterial.enableInstancing = true;
        
        Graphics.DrawMeshInstancedIndirect(
            sphereMesh, 0, sphereMaterial,
            new Bounds(Vector3.zero, Vector3.one * spawnRadius * 2f),
            drawArgsBufferSpheres);
        UpdateDragVisualization();
        UpdateParticleLabels();  // refresh labels
        UpdateSplitPlaneRings(); // update split plane rings
    }    // UpdateAdhesionConnectionsVisual method has been removed as part of bond removal

    #endregion
    
    #region Event Handlers
    
    void OnGenomeChanged()
    {
        // Update the genome modes buffer when the genome changes
        UpdateGenomeModesBuffer();
        
        // Reinitialize particles when genome changes
        if (particleBuffer != null)
        {
            InitializeParticles();
        }
    }
    
    #endregion

    #region Initialization and Cleanup
    
    private void InitializeBuffers()
    {
        int stride = 84; // Updated from 80 to 84 to include the modeIndex field
        particleBuffer = new ComputeBuffer(particleCount, stride);
        dragInputBuffer = new ComputeBuffer(1, sizeof(int) + sizeof(float) * 4);
        positionReadbackBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        rotationReadbackBuffer = new ComputeBuffer(particleCount, sizeof(float) * 4);
        drawArgsBufferSpheres = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);

        gridHeads = new ComputeBuffer(GRID_TOTAL, sizeof(uint));
        gridNext = new ComputeBuffer(particleCount, sizeof(uint));
        gridParticleIndices = new ComputeBuffer(particleCount, sizeof(uint));
        torqueAccumBuffer = new ComputeBuffer(particleCount, sizeof(int) * 3);

        cpuParticlePositions = new Vector3[particleCount];
        cpuParticleRotations = new Quaternion[particleCount];

        kernelInitParticles     = computeShader.FindKernel("InitParticles");
        kernelClearGrid         = computeShader.FindKernel("ClearGrid");
        kernelBuildGrid         = computeShader.FindKernel("BuildHashGrid");
        kernelApplySPHForces    = computeShader.FindKernel("ApplySPHForces");
        kernelApplyDrag         = computeShader.FindKernel("ApplyDragForce");
        kernelUpdateMotion      = computeShader.FindKernel("UpdateMotion");
        kernelUpdateRotation    = computeShader.FindKernel("UpdateRotation");        kernelCopyPositions     = computeShader.FindKernel("CopyPositionsToReadbackBuffer");
        kernelCopyRotations     = computeShader.FindKernel("CopyRotationsToReadbackBuffer");
        kernelApplyAdhesionConstraints = computeShader.FindKernel("ApplyAdhesionConstraints");
        kernelApplyAdhesionDeltas = computeShader.FindKernel("ApplyAdhesionDeltas");

        uint[] args = new uint[5]
        {
            sphereMesh.GetIndexCount(0),
            (uint)particleCount,
            sphereMesh.GetIndexStart(0),
            sphereMesh.GetBaseVertex(0),
            0
        };
        drawArgsBufferSpheres.SetData(args);

        // Initialize LineRenderers
        GameObject circleObject = new GameObject("DragCircle");
        circleRenderer = circleObject.AddComponent<LineRenderer>();
        circleRenderer.startWidth = 0.02f;
        circleRenderer.endWidth = 0.02f;
        circleRenderer.loop = true;
        circleRenderer.positionCount = 36;
        circleRenderer.material = new Material(Shader.Find("Sprites/Default"));
        circleRenderer.material.color = dragCircleColor;
        circleRenderer.enabled = false;

        GameObject lineObject = new GameObject("DragLine");
        lineRenderer = lineObject.AddComponent<LineRenderer>();
        lineRenderer.startWidth = 0.02f;
        lineRenderer.endWidth = 0.02f;
        lineRenderer.positionCount = 2;
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.material.color = dragCircleColor;
        lineRenderer.enabled = false;

        // --- Ensure adhesionConnectionBuffer is allocated ---
        if (adhesionConnectionBuffer != null)
        {
            adhesionConnectionBuffer.Release();
        }        // Each bond is a struct with updated size: int, int, float, float, float, float4, float4, float3, float3, float, int
        // = 2*4 + 3*4 + 4*4 + 4*4 + 3*4 + 3*4 + 4 + 4 = 8 + 12 + 16 + 16 + 12 + 12 + 4 + 4 = 84 bytes
        adhesionConnectionBuffer = new ComputeBuffer(maxAdhesionConnections, 84);

        // --- Ensure adhesionVelocityDeltaBuffer and adhesionRotationDeltaBuffer are allocated ---
        if (adhesionVelocityDeltaBuffer != null)
        {
            adhesionVelocityDeltaBuffer.Release();
        }
        adhesionVelocityDeltaBuffer = new ComputeBuffer(particleCount * 3, sizeof(int));

        if (adhesionRotationDeltaBuffer != null)
        {
            adhesionRotationDeltaBuffer.Release();
        }
        adhesionRotationDeltaBuffer = new ComputeBuffer(particleCount * 4, sizeof(int));
    }

    private void ReleaseBuffers()
    {
        particleBuffer?.Release();
        dragInputBuffer?.Release();
        drawArgsBufferSpheres?.Release();
        positionReadbackBuffer?.Release();
        rotationReadbackBuffer?.Release();
        gridHeads?.Release();
        gridNext?.Release();
        gridParticleIndices?.Release();
        torqueAccumBuffer?.Release();
        genomeModesBuffer?.Release(); // Release the genome modes buffer
        adhesionConnectionBuffer?.Release();
        adhesionVelocityDeltaBuffer?.Release();
        adhesionRotationDeltaBuffer?.Release();
        
        if (circleRenderer != null) Destroy(circleRenderer.gameObject);
        if (lineRenderer != null) Destroy(lineRenderer.gameObject);
        
        if (particleLabels != null)
        {
            foreach (var lbl in particleLabels)
            {
                if (lbl != null)
                {
                    Destroy(lbl.gameObject);
                }
            }
        }
    }

    private void InitializeParticles()
    {        // Initialize particle IDs array
        if (particleIDs == null || particleIDs.Length != particleCount)
        {
            particleIDs = new ParticleIDData[particleCount];
            
            // Set initial particle (root cell) with ID 00.00.A
            particleIDs[0].parentID = 0;
            particleIDs[0].uniqueID = 0;
            particleIDs[0].childType = 'A';
        }

        computeShader.SetFloat("spawnRadius", spawnRadius);
        computeShader.SetFloat("minRadius", minRadius);
        computeShader.SetFloat("maxRadius", maxRadius);
        computeShader.SetFloat("torqueFactor", torqueFactor);
        computeShader.SetFloat("torqueDamping", torqueDamping);
        computeShader.SetFloat("boundaryFriction", boundaryFriction);
        computeShader.SetFloat("rollingContactRadiusMultiplier", rollingContactRadiusMultiplier);
        computeShader.SetFloat("density", density);
        computeShader.SetFloat("repulsionStrength", repulsionStrength);

        int initialParticleCount = 1; // Start with just a single particle

        computeShader.SetInt("activeParticleCount", initialParticleCount);

        computeShader.SetBuffer(kernelInitParticles, "particleBuffer", particleBuffer);
        computeShader.SetBuffer(kernelInitParticles, "torqueAccumBuffer", torqueAccumBuffer);
        computeShader.Dispatch(kernelInitParticles, Mathf.CeilToInt(particleCount / 64f), 1, 1);

        // Explicitly set the initial mode for the first particle
        if (genome != null && genome.modes.Count > 0)
        {
            int initialModeIndex = GetInitialModeIndex();
            Particle[] firstParticle = new Particle[1];
            particleBuffer.GetData(firstParticle, 0, 0, 1);
            firstParticle[0].modeIndex = initialModeIndex;

            particleBuffer.SetData(firstParticle, 0, 0, 1);
        }

        if (cellSplitTimers == null || cellSplitTimers.Length < particleCount)
        {
            cellSplitTimers = new float[particleCount];
        }

        for (int i = 0; i < particleCount; i++)
        {
            if (genome != null && i < activeParticleCount)
            {
                Particle[] particleData = new Particle[1];
                particleBuffer.GetData(particleData, 0, i, 1);

                int modeIndex = particleData[0].modeIndex;
                if (modeIndex >= 0 && modeIndex < genome.modes.Count)
                {
                    cellSplitTimers[i] = 0f;
                }
                else
                {
                    cellSplitTimers[i] = 0f;
                }
            }
            else
            {
                cellSplitTimers[i] = 0f;
            }
        }
    }

    private void InitializeParticleLabels()
    {
        if (particleLabels == null || particleLabels.Length != particleCount)
        {
            // clean up old labels
            if (particleLabels != null)
            {
                foreach (var lbl in particleLabels) if (lbl != null) Destroy(lbl.gameObject);
            }
            GameObject labelsParent = new GameObject("ParticleLabels");
            labelsParent.transform.SetParent(transform);
            particleLabels = new TextMeshPro[particleCount];
            
            // Only create actual GameObjects for active particles
            // This saves memory and keeps the hierarchy clean
            for (int i = 0; i < activeParticleCount; i++)
            {
                // Create label name using particle ID if available, otherwise use index
                string labelName;
                if (particleIDs != null && i < particleIDs.Length && i < activeParticleCount)
                {
                    labelName = $"Label_{particleIDs[i].GetFormattedID()}";
                }
                else
                {
                    labelName = $"Label_{i}";
                }
                
                GameObject labelObj = new GameObject(labelName);
                labelObj.transform.SetParent(labelsParent.transform);
                var tmp = labelObj.AddComponent<TextMeshPro>();
                tmp.alignment = TextAlignmentOptions.Center;
                tmp.fontSize = 3;
                tmp.color = labelColor;
                tmp.transform.localScale = Vector3.one * labelScale;
                tmp.enabled = false;
                particleLabels[i] = tmp;
            }
        }
    }

    private void InitializeSplitPlaneRings()
    {
        if (splitPlaneRings == null || splitPlaneRings.Length != particleCount)
        {
            // Clean up old rings
            if (splitPlaneRings != null)
            {
                foreach (var ring in splitPlaneRings)
                {
                    if (ring != null) Destroy(ring.gameObject);
                }
            }
            splitPlaneRings = new LineRenderer[particleCount];
            GameObject ringsParent = new GameObject("SplitPlaneRings");
            ringsParent.transform.SetParent(transform);
            for (int i = 0; i < activeParticleCount; i++)
            {
                GameObject ringObj = new GameObject($"SplitPlaneRing_{i}");
                ringObj.transform.SetParent(ringsParent.transform);
                var lr = ringObj.AddComponent<LineRenderer>();
                lr.positionCount = splitPlaneRingSegments + 1;
                lr.loop = true;
                lr.startWidth = 0.03f;
                lr.endWidth = 0.03f;
                lr.material = new Material(Shader.Find("Sprites/Default"));
                lr.material.color = splitPlaneRingColor;
                lr.enabled = false;
                splitPlaneRings[i] = lr;
            }
        }
    }
    
    #endregion

    #region Cell Division and Management
    
    private void UpdateCellDivisionTimers(float deltaTime)
    {
        if (cellSplitTimers == null || cellSplitTimers.Length < activeParticleCount)
        {
            float[] newTimers = new float[particleCount];
            if (cellSplitTimers != null)
            {
                System.Array.Copy(cellSplitTimers, newTimers, cellSplitTimers.Length);
            }
            cellSplitTimers = newTimers;
        }
        
        if (pendingSplits.Count > 0)
        {
            ProcessPendingSplits();
        }
        
        int allowedSplits = particleCount - activeParticleCount;
        if (allowedSplits <= 0 || genome == null || genome.modes.Count == 0) return;
        
        // Update all timers
        for (int i = 0; i < activeParticleCount; i++)
        {
            cellSplitTimers[i] += deltaTime;
        }
        
        // Create a list to collect cells that are ready to split
        List<int> cellsReadyToSplit = new List<int>();
        const float epsilon = 0.001f;

        // First pass: identify all cells ready to split
        if (cachedParticleDataValid)
        {
            // Use cached data if available
            for (int i = 0; i < activeParticleCount && i < cachedParticleData.Length; i++)
            {
                int modeIndex = cachedParticleData[i].modeIndex;
                
                if (modeIndex >= 0 && modeIndex < genome.modes.Count)
                {
                    float splitInterval = genome.modes[modeIndex].splitInterval;
                    
                    if (cellSplitTimers[i] >= splitInterval - epsilon)
                    {
                        // Add to split list if we still have room for new particles
                        if (cellsReadyToSplit.Count < allowedSplits)
                        {
                            cellsReadyToSplit.Add(i);
                        }
                        
                        // Reset timer regardless of whether we can actually split now
                        cellSplitTimers[i] = 0f;
                    }
                }
            }
        }
        else
        {
            // If cached data is not available, read directly from GPU buffer
            Particle[] particleData = new Particle[activeParticleCount];
            particleBuffer.GetData(particleData, 0, 0, activeParticleCount);
            
            for (int i = 0; i < activeParticleCount; i++)
            {
                int modeIndex = particleData[i].modeIndex;
                
                if (modeIndex >= 0 && modeIndex < genome.modes.Count)
                {
                    float splitInterval = genome.modes[modeIndex].splitInterval;
                    
                    if (cellSplitTimers[i] >= splitInterval - epsilon)
                    {
                        // Add to split list if we still have room for new particles
                        if (cellsReadyToSplit.Count < allowedSplits)
                        {
                            cellsReadyToSplit.Add(i);
                        }
                        
                        // Reset timer regardless of whether we can actually split now
                        cellSplitTimers[i] = 0f;
                    }
                }
            }
        }
        
        // Second pass: process all splits at once
        foreach (int cellIndex in cellsReadyToSplit)
        {
            SplitCell(cellIndex);
        }
        
        // Reset cache validity flag
        if (cachedParticleDataValid && cellsReadyToSplit.Count > 0)
        {
            cachedParticleDataValid = false;
        }
    }

    private void SplitCell(int parentIndex)
    {
        if (genome == null || genome.modes.Count == 0 || parentIndex >= activeParticleCount)
            return;
        Vector3 parentPos = cpuParticlePositions[parentIndex];
        Quaternion parentRot = cpuParticleRotations[parentIndex];
        Particle[] particleData = new Particle[1];
        particleBuffer.GetData(particleData, 0, parentIndex, 1);
        int parentModeIndex = particleData[0].modeIndex;
        if (parentModeIndex < 0 || parentModeIndex >= genome.modes.Count) {
            parentModeIndex = GetInitialModeIndex();
        }
        GenomeMode mode = genome.modes[parentModeIndex];
        int childAModeIndex = mode.childAModeIndex;
        if (childAModeIndex < 0 || childAModeIndex >= genome.modes.Count)
            childAModeIndex = parentModeIndex;
        int childBModeIndex = mode.childBModeIndex;
        if (childBModeIndex < 0 || childBModeIndex >= genome.modes.Count)
            childBModeIndex = parentModeIndex;
        Vector3 forward = parentRot * Vector3.forward;
        Vector3 up = parentRot * Vector3.up;
        Vector3 right = parentRot * Vector3.right;
        Vector3 splitDirLocal = GetDirection(mode.parentSplitYaw, mode.parentSplitPitch);
        Vector3 splitDirWorld = right * splitDirLocal.x + up * splitDirLocal.y + forward * splitDirLocal.z;
        Vector3 posA = parentPos + splitDirWorld * spawnOverlapOffset;
        Vector3 posB = parentPos - splitDirWorld * spawnOverlapOffset;
        Vector3 childADirLocal = GetDirection(mode.childA_OrientationYaw, mode.childA_OrientationPitch);
        Vector3 childADirWorld = right * childADirLocal.x + up * childADirLocal.y + forward * childADirLocal.z;
        Quaternion rotA = Quaternion.LookRotation(childADirWorld, up);
        Vector3 childBDirLocal = GetDirection(mode.childB_OrientationYaw, mode.childB_OrientationPitch);
        Vector3 childBDirWorld = right * childBDirLocal.x + up * childBDirLocal.y + forward * childBDirLocal.z;
        Quaternion rotB = Quaternion.LookRotation(childBDirWorld, up);
        Vector3 parentVelocity = Vector3.zero;
        Vector3 velA = parentVelocity + splitDirWorld * splitVelocityMagnitude;
        Vector3 velB = parentVelocity - splitDirWorld * splitVelocityMagnitude;
        int parentUniqueID = particleIDs[parentIndex].uniqueID;
        CellSplitData splitData = new CellSplitData
        {
            parentIndex = parentIndex,
            positionA = posA,
            positionB = posB,
            velocityA = velA,
            velocityB = velB,
            rotationA = rotA,
            rotationB = rotB,
            childAModeIndex = childAModeIndex,
            childBModeIndex = childBModeIndex
        };
        pendingSplits.Add(splitData);
    }
    
    private void ProcessPendingSplits()
    {
        if (pendingSplits.Count == 0)
            return;

        // Calculate how many new particles we'll need (only +1 per split since we're reusing parent's slot)
        int neededNewParticles = pendingSplits.Count;
        // Check if there's enough room for the new particles
        if (activeParticleCount + neededNewParticles > particleCount)
        {
            int newCapacity = Mathf.Max(activeParticleCount + neededNewParticles, particleCount * 2);
            ResizeParticleBuffers(newCapacity);
        }
        Particle[] particleData = new Particle[particleCount];
        particleBuffer.GetData(particleData);
        if (particleIDs == null || particleIDs.Length < particleCount)
        {
            ParticleIDData[] newParticleIDs = new ParticleIDData[particleCount];
            if (particleIDs != null)
            {
                System.Array.Copy(particleIDs, newParticleIDs, System.Math.Min(particleIDs.Length, particleCount));
            }
            particleIDs = newParticleIDs;
            if (activeParticleCount > 0 && particleIDs[0].childType == '\0')
            {
                particleIDs[0].parentID = 0;
                particleIDs[0].uniqueID = 0;
                particleIDs[0].childType = 'A';
            }
        }
        // Ensure splitPlaneRings array is large enough
        if (splitPlaneRings == null || splitPlaneRings.Length < particleCount)
        {
            LineRenderer[] newRings = new LineRenderer[particleCount];
            if (splitPlaneRings != null)
            {
                System.Array.Copy(splitPlaneRings, newRings, System.Math.Min(splitPlaneRings.Length, particleCount));
            }
            splitPlaneRings = newRings;
        }
        // Find or create the rings parent object
        Transform ringsParent = transform.Find("SplitPlaneRings");
        if (ringsParent == null)
        {
            GameObject ringsParentObj = new GameObject("SplitPlaneRings");
            ringsParentObj.transform.SetParent(transform);
            ringsParent = ringsParentObj.transform;
        }
        // Keep track of newly created particles so we can create labels for them later
        List<int> newParticleIndices = new List<int>();
        List<string> newParticleIds = new List<string>();

        foreach (var split in pendingSplits)
        {
            int parentIndex = split.parentIndex;
            
            // Verify parent index is valid before accessing arrays
            if (parentIndex < 0 || parentIndex >= particleCount)
            {
                Debug.LogWarning($"Invalid parent index: {parentIndex}. Skipping this split.");
                continue;
            }
            
            // Store parent's unique ID for children to reference
            int originalParentID = particleIDs[parentIndex].uniqueID;
            
            // We'll reuse the parent's slot for Child A, and add Child B at the end
            int childAIndex = parentIndex; // Reuse parent's slot
            int childBIndex = activeParticleCount; // Add at the end
              // Use global counter for uniqueIDs to ensure they're never reused
            int childAUniqueID = nextUniqueIDCounter++;
            int childBUniqueID = nextUniqueIDCounter++;

            // Overwrite parent with Child A
            particleData[childAIndex].position = split.positionA;
            particleData[childAIndex].velocity = split.velocityA;
            particleData[childAIndex].rotation = split.rotationA;
            particleData[childAIndex].modeIndex = split.childAModeIndex;
            
            // Set Child A's ID - using original parent's unique ID as the parent ID
            particleIDs[childAIndex].parentID = originalParentID;
            particleIDs[childAIndex].uniqueID = childAUniqueID;
            particleIDs[childAIndex].childType = 'A';

            // Create Child B (copy from parent data)
            particleData[childBIndex] = particleData[childAIndex]; 
            particleData[childBIndex].position = split.positionB;
            particleData[childBIndex].velocity = split.velocityB;
            particleData[childBIndex].rotation = split.rotationB;
            particleData[childBIndex].modeIndex = split.childBModeIndex;
            
            // Set Child B's ID - using original parent's unique ID as the parent ID
            particleIDs[childBIndex].parentID = originalParentID;
            particleIDs[childBIndex].uniqueID = childBUniqueID;
            particleIDs[childBIndex].childType = 'B';
            
            // Initialize split timers for the new cells
            if (cellSplitTimers == null || cellSplitTimers.Length < particleCount)
            {
                float[] newSplitTimers = new float[particleCount];
                if (cellSplitTimers != null)
                {
                    System.Array.Copy(cellSplitTimers, newSplitTimers, System.Math.Min(cellSplitTimers.Length, particleCount));
                }
                cellSplitTimers = newSplitTimers;
            }
            
            cellSplitTimers[childAIndex] = 0f;
            cellSplitTimers[childBIndex] = 0f;
            
            // Track the new particles for label creation
            // Child A replaces the parent, so we need a label for it
            newParticleIndices.Add(childAIndex);
            newParticleIds.Add(particleIDs[childAIndex].GetFormattedID());
            
            // Child B is completely new and needs a label
            newParticleIndices.Add(childBIndex);
            newParticleIds.Add(particleIDs[childBIndex].GetFormattedID());
            
            // Ensure split rings exist for both children
            if (splitPlaneRings[childAIndex] == null)
            {
                GameObject ringObj = new GameObject($"SplitPlaneRing_{childAIndex}");
                ringObj.transform.SetParent(ringsParent);
                var lr = ringObj.AddComponent<LineRenderer>();
                lr.positionCount = splitPlaneRingSegments + 1;
                lr.loop = true;
                lr.startWidth = 0.03f;
                lr.endWidth = 0.03f;
                lr.material = new Material(Shader.Find("Sprites/Default"));
                lr.material.color = splitPlaneRingColor;
                lr.enabled = false;
                splitPlaneRings[childAIndex] = lr;
            }
            if (splitPlaneRings[childBIndex] == null)
            {
                GameObject ringObj = new GameObject($"SplitPlaneRing_{childBIndex}");
                ringObj.transform.SetParent(ringsParent);
                var lr = ringObj.AddComponent<LineRenderer>();
                lr.positionCount = splitPlaneRingSegments + 1;
                lr.loop = true;
                lr.startWidth = 0.03f;
                lr.endWidth = 0.03f;
                lr.material = new Material(Shader.Find("Sprites/Default"));
                lr.material.color = splitPlaneRingColor;
                lr.enabled = false;
                splitPlaneRings[childBIndex] = lr;
            }
            
            // --- ADHESION SYSTEM INTEGRATION ---
            if (adhesionManager != null && genome != null)
            {
                // Get parent mode for split parameters
                int parentModeIndex = particleData[parentIndex].modeIndex;
                if (parentModeIndex < 0 || parentModeIndex >= genome.modes.Count)
                    parentModeIndex = 0;
                var mode = genome.modes[parentModeIndex];
                adhesionManager.HandleCellSplit(
                    parentIndex,
                    childAIndex,
                    childBIndex,
                    mode.parentSplitYaw,
                    mode.parentSplitPitch,
                    cpuParticleRotations[parentIndex],
                    cpuParticlePositions[parentIndex],
                    split.childAModeIndex,
                    split.childBModeIndex,
                    mode.parentMakeAdhesion,
                    mode.childA_KeepAdhesion,
                    mode.childB_KeepAdhesion
                );
            }
            // --- END ADHESION SYSTEM INTEGRATION ---
            
            // Increment active count by only 1 since we're reusing the parent's slot
            activeParticleCount += 1;
        }

        // Write updated particle data back to the buffer
        particleBuffer.SetData(particleData);
          // Clear pending splits after processing
        pendingSplits.Clear();
          // Create labels for the new particles
        CreateLabelsForNewParticles(newParticleIndices, newParticleIds);
    }

    private Vector3 GetDirection(float yaw, float pitch)
    {
        return Quaternion.Euler(pitch, yaw, 0f) * Vector3.forward;
    }
    
    #endregion

    #region Particle Interaction
    
    void HandleMouseDrag()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            float closestDist = Mathf.Infinity;
            int closestID = -1;
            for (int i = 0; i < activeParticleCount; i++) // Only check active particles
            {
                Vector3 center = cpuParticlePositions[i];
                float r = maxRadius;
                Vector3 oc = ray.origin - center;
                float a = Vector3.Dot(ray.direction, ray.direction);
                float b = 2.0f * Vector3.Dot(oc, ray.direction);
                float c = Vector3.Dot(oc, oc) - r * r;
                float discriminant = b * b - 4.0f * a * c;
                if (discriminant > 0)
                {
                    float t = (-b - Mathf.Sqrt(discriminant)) / (2.0f * a);
                    if (t > 0 && t < closestDist)
                    {
                        closestID = i;
                        closestDist = t;
                    }
                }
            }
            if (closestID != -1)
            {
                selectedParticleID = closestID;
                lastSelectedParticleID = closestID;
                dragTargetWorld = cpuParticlePositions[selectedParticleID];
                currentDragDistance = Vector3.Distance(Camera.main.transform.position, dragTargetWorld);
                // Display additional info in the console
                string idText = (particleIDs != null && selectedParticleID < particleIDs.Length) ? particleIDs[selectedParticleID].GetFormattedID() : selectedParticleID.ToString();
                string modeText = (cachedParticleData != null && selectedParticleID < cachedParticleData.Length) ? $"Mode: {cachedParticleData[selectedParticleID].modeIndex}" : "Mode: N/A";
                Vector3 pos = cpuParticlePositions[selectedParticleID];
                Debug.Log($"Selected Particle: {idText}, Index: {selectedParticleID}, {modeText}, Position: {pos}");
                UpdateSplitPlaneRings(); // Ensure split ring updates immediately
            }
        }

        if (Input.GetMouseButton(0) && selectedParticleID != -1)
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            dragTargetWorld = Camera.main.transform.position + ray.direction * currentDragDistance;
        }

        if (Input.GetMouseButtonUp(0))
        {
            selectedParticleID = -1;
        }

        DragInput drag = new DragInput
        {
            selectedID = selectedParticleID,
            targetPosition = dragTargetWorld,
            strength = Input.GetMouseButton(0) ? 100f : 0f
        };
        dragInputBuffer.SetData(new DragInput[] { drag });
    }

    private void UpdateDragVisualization()
    {
        if (selectedParticleID != -1)
        {
            // Update circle
            Vector3 cameraForward = Camera.main.transform.forward;
            Vector3 cameraRight = Camera.main.transform.right;
            Vector3 cameraUp = Vector3.Cross(cameraForward, cameraRight);

            for (int i = 0; i < 36; i++)
            {
                float angle = Mathf.Deg2Rad * (i * 10);
                Vector3 point = dragTargetWorld + (Mathf.Cos(angle) * cameraRight + Mathf.Sin(angle) * cameraUp) * dragCircleRadius;
                circleRenderer.SetPosition(i, point);
            }
            circleRenderer.enabled = true;

            // Update line
            lineRenderer.SetPosition(0, cpuParticlePositions[selectedParticleID]);
            lineRenderer.SetPosition(1, dragTargetWorld);
            lineRenderer.enabled = true;
        }
        else
        {
            circleRenderer.enabled = false;
            lineRenderer.enabled = false;
        }
    }
    
    private void UpdateSplitPlaneRings()
    {
        if (!showSplitPlaneRings || splitPlaneRings == null)
        {
            if (splitPlaneRings != null)
                foreach (var ring in splitPlaneRings) if (ring != null) ring.enabled = false;
            return;
        }
        if (Camera.main == null) return;
        // Only render the split ring for the last selected cell
        for (int i = 0; i < splitPlaneRings.Length; i++)
        {
            if (splitPlaneRings[i] != null) splitPlaneRings[i].enabled = false;
        }
        if (lastSelectedParticleID >= 0 && lastSelectedParticleID < activeParticleCount && splitPlaneRings[lastSelectedParticleID] != null)
        {
            int i = lastSelectedParticleID;
            var lr = splitPlaneRings[i];
            Vector3 center = cpuParticlePositions[i];
            Quaternion rot = cpuParticleRotations[i];
            Vector3 normal = rot * Vector3.up; // fallback
            if (cachedParticleData != null && i < cachedParticleData.Length && genome != null && cachedParticleData[i].modeIndex >= 0 && cachedParticleData[i].modeIndex < genome.modes.Count)
            {
                var mode = genome.modes[cachedParticleData[i].modeIndex];
                Vector3 splitDirLocal = GetDirection(mode.parentSplitYaw, mode.parentSplitPitch);
                Vector3 right = rot * Vector3.right;
                Vector3 up = rot * Vector3.up;
                Vector3 forward = rot * Vector3.forward;
                Vector3 splitDirWorld = right * splitDirLocal.x + up * splitDirLocal.y + forward * splitDirLocal.z;
                normal = splitDirWorld;
            }
            Vector3[] points = new Vector3[splitPlaneRingSegments + 1];
            Quaternion ringRot = Quaternion.FromToRotation(Vector3.up, normal);
            for (int j = 0; j <= splitPlaneRingSegments; j++)
            {
                float angle = Mathf.Deg2Rad * (j * 360f / splitPlaneRingSegments);
                Vector3 localPos = new Vector3(Mathf.Cos(angle) * splitPlaneRingRadius, 0, Mathf.Sin(angle) * splitPlaneRingRadius);
                points[j] = center + ringRot * localPos;
            }
            lr.SetPositions(points);
            lr.startColor = splitPlaneRingColor;
            lr.endColor = splitPlaneRingColor;
            lr.enabled = true;
        }
    }

    #endregion
    
    #region GPU Data Management
    
    private void RequestParticleDataAsync()
    {
        if (!particleDataReadbackInProgress && !cachedParticleDataValid)
        {
            particleDataReadbackInProgress = true;
            
            AsyncGPUReadback.Request(particleBuffer, r => 
            {
                if (r.hasError)
                {
                    Debug.LogError("AsyncGPUReadback error: Failed to read particle data");
                    particleDataReadbackInProgress = false;
                    return;
                }
                
                if (cachedParticleData == null || cachedParticleData.Length < activeParticleCount)
                {
                    cachedParticleData = new Particle[particleCount];
                }
                
                r.GetData<Particle>().CopyTo(cachedParticleData);
                  // Set cached data as valid
                cachedParticleDataValid = true;
                particleDataReadbackInProgress = false;
            });
        }
        
        // Request position data
        AsyncGPUReadback.Request(positionReadbackBuffer, r => 
        {
            if (!r.hasError)
            {
                r.GetData<Vector3>().CopyTo(cpuParticlePositions);
            }
        });
        
        // Request rotation data
        AsyncGPUReadback.Request(rotationReadbackBuffer, r => 
        {
            if (!r.hasError)
            {
                r.GetData<Quaternion>().CopyTo(cpuParticleRotations);
            }
        });
    }

    // Resize particle buffers when more capacity is needed
    private void ResizeParticleBuffers(int newCapacity)
    {
        // Store old data
        Particle[] oldParticleData = new Particle[particleCount];
        particleBuffer.GetData(oldParticleData);
        
        ParticleIDData[] oldIDData = new ParticleIDData[particleCount];
        if (particleIDs != null)
        {
            System.Array.Copy(particleIDs, oldIDData, particleCount);
        }
        
        float[] oldSplitTimers = new float[particleCount];
        if (cellSplitTimers != null)
        {
            System.Array.Copy(cellSplitTimers, oldSplitTimers, particleCount);
        }
        
        // Release old buffers
        ReleaseBuffers();
        
        // Update capacity
        particleCount = newCapacity;
        
        // Reinitialize buffers with new capacity
        InitializeBuffers();
        
        // Restore old data
        Particle[] newParticleData = new Particle[particleCount];
        System.Array.Copy(oldParticleData, newParticleData, oldParticleData.Length);
        particleBuffer.SetData(newParticleData);
          // Restore particle IDs
        particleIDs = new ParticleIDData[particleCount];
        System.Array.Copy(oldIDData, particleIDs, oldIDData.Length);
        
        // Restore split timers
        cellSplitTimers = new float[particleCount];
        System.Array.Copy(oldSplitTimers, cellSplitTimers, oldSplitTimers.Length);
        
        // Copy particle positions and rotations from CPU arrays to the new larger arrays
        Vector3[] newPositions = new Vector3[particleCount];
        Quaternion[] newRotations = new Quaternion[particleCount];
        
        if (cpuParticlePositions != null)
        {
            System.Array.Copy(cpuParticlePositions, newPositions, System.Math.Min(cpuParticlePositions.Length, particleCount));
        }
        
        if (cpuParticleRotations != null)
        {
            System.Array.Copy(cpuParticleRotations, newRotations, System.Math.Min(cpuParticleRotations.Length, particleCount));
        }
        
        cpuParticlePositions = newPositions;
        cpuParticleRotations = newRotations;
        
        // Reinitialize particle labels with new count
        InitializeParticleLabels();
        // Reinitialize split plane rings with new count
        InitializeSplitPlaneRings();
    }

    private void UpdateGenomeModesBuffer()
    {
        if (genome == null || genome.modes.Count == 0)
        {
            Debug.LogWarning("Cannot update genome modes buffer: No genome or modes available");
            return;
        }

        // Release existing buffer if any
        if (genomeModesBuffer != null)
        {
            genomeModesBuffer.Release();
            genomeModesBuffer = null;
        }

        int modeCount = genome.modes.Count;
        genomeModesBuffer = new ComputeBuffer(modeCount, System.Runtime.InteropServices.Marshal.SizeOf(typeof(GenomeColorData)));

        GenomeColorData[] modeData = new GenomeColorData[modeCount];
        for (int i = 0; i < modeCount; i++)
        {
            GenomeMode mode = genome.modes[i];
            uint colorPacked = PackColorToUint(mode.modeColor);
            modeData[i] = new GenomeColorData
            {
                parentMakeAdhesion = 0, // Set as needed
                childA_KeepAdhesion = 0, // Set as needed
                childB_KeepAdhesion = 0, // Set as needed
                adhesionRestLength = 0f, // Set as needed
                adhesionSpringStiffness = 0f, // Set as needed
                adhesionSpringDamping = 0f, // Set as needed
                colorPacked = colorPacked,
                orientConstraintStrength = mode.orientationConstraintStrength,
                maxAngleDeviation = mode.maxAllowedAngleDeviation
            };
        }
        genomeModesBuffer.SetData(modeData);
        sphereMaterial.SetBuffer("genomeModesBuffer", genomeModesBuffer);
        sphereMaterial.SetInt("defaultGenomeMode", GetInitialModeIndex());
    }
    
    // Utility function to pack a Color into a uint
    private uint PackColorToUint(Color color)
    {
        uint r = (uint)(color.r * 255);
        uint g = (uint)(color.g * 255);
        uint b = (uint)(color.b * 255);
        return (r << 16) | (g << 8) | b;
    }

    private int GetInitialModeIndex()
    {
        if (genome == null || genome.modes.Count == 0)
            return 0;
            
        for (int i = 0; i < genome.modes.Count; i++)
        {
            if (genome.modes[i].isInitial)
                return i;
        }
        
        return 0; // Default to first mode if no initial mode is marked
    }
    
    #endregion

    #region UI and Visualization
    
    private void UpdateParticleLabels()
    {
        if (!showParticleLabels)
        {
            if (particleLabels != null)
                foreach (var lbl in particleLabels) if (lbl != null) lbl.enabled = false;
            return;
        }
        if (Camera.main == null) return;

        int count = Mathf.Min(activeParticleCount, particleLabels.Length);
        for (int i = 0; i < count; i++)
        {
            var tmp = particleLabels[i];
            if (tmp == null) continue;
            
            // Validate particle position before using it
            Vector3 particlePos = cpuParticlePositions[i];
            if (float.IsNaN(particlePos.x) || float.IsNaN(particlePos.y) || float.IsNaN(particlePos.z))
            {
                tmp.enabled = false;
                continue; // Skip this label if the position contains NaN
            }
            
            // Position labels slightly lower than before by adding a small negative Y offset
            Vector3 pos = particlePos + (Vector3.up * labelOffset * 0.5f);
            
            // Additional validation before setting the position
            if (float.IsNaN(pos.x) || float.IsNaN(pos.y) || float.IsNaN(pos.z))
            {
                tmp.enabled = false;
                continue;
            }
            
            tmp.transform.position = pos;
            tmp.transform.rotation = Camera.main.transform.rotation;
            
            // Display the formatted ID
            string text = "";
            if (particleIDs != null && i < particleIDs.Length) 
            {
                text = particleIDs[i].GetFormattedID();
                
                // Update GameObject name to use the formatted ID for better hierarchy organization
                string labelName = $"Label_{text}";
                if (tmp.gameObject.name != labelName)
                {
                    tmp.gameObject.name = labelName;
                }
            }
            
            tmp.text = text;
            tmp.enabled = true;
        }
        for (int i = count; i < particleLabels.Length; i++)
        {
            if (particleLabels[i] != null) particleLabels[i].enabled = false;
        }
    }
    
    private void CreateLabelsForNewParticles(List<int> indices, List<string> ids)
    {
        if (indices == null || ids == null || indices.Count != ids.Count)
        {
            Debug.LogWarning("Invalid input to CreateLabelsForNewParticles.");
            return;
        }

        // Find or create the labels parent object
        Transform labelsParent = transform.Find("ParticleLabels");
        if (labelsParent == null)
        {
            GameObject labelsParentObj = new GameObject("ParticleLabels");
            labelsParentObj.transform.SetParent(transform);
            labelsParent = labelsParentObj.transform;
        }

        for (int i = 0; i < indices.Count; i++)
        {
            int index = indices[i];
            string id = ids[i];

            if (index < 0 || index >= particleLabels.Length)
            {
                Debug.LogWarning($"Invalid index {index} for new particle label.");
                continue;
            }

            // We may already have a label at this index (especially for Child A which replaces parent)
            // So we need to either update it or create a new one
            if (particleLabels[index] != null)
            {
                particleLabels[index].gameObject.name = $"Label_{id}";
            }
            else
            {
                GameObject labelObj = new GameObject($"Label_{id}");
                labelObj.transform.SetParent(labelsParent);
                var tmp = labelObj.AddComponent<TextMeshPro>();
                tmp.alignment = TextAlignmentOptions.Center;
                tmp.fontSize = 3;
                tmp.color = labelColor;
                tmp.transform.localScale = Vector3.one * labelScale;
                tmp.enabled = false;
                particleLabels[index] = tmp;
            }
        }
    }
    
    #endregion
}