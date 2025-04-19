using UnityEngine;
using System.Collections.Generic;
using UnityEngine.Rendering;  // Add this for AsyncGPUReadback

public class ParticleSystemController : MonoBehaviour
{
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
    [Range(0f, 10f)] public float density = 1.0f;
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

    [Header("Adhesion Visualization")]
    public bool showAdhesionConnections = true;
    public float adhesionLineWidth = 0.2f; // Increased from 0.05f to make lines more visible

    private LineRenderer circleRenderer;
    private LineRenderer lineRenderer;

    ComputeBuffer particleBuffer, dragInputBuffer, drawArgsBufferSpheres;
    ComputeBuffer positionReadbackBuffer, rotationReadbackBuffer;
    ComputeBuffer gridHeads, gridNext, gridParticleIndices;
    ComputeBuffer torqueAccumBuffer;

    const int GRID_DIM = 32;
    const int GRID_TOTAL = GRID_DIM * GRID_DIM * GRID_DIM;

    int kernelInitParticles;
    int kernelClearGrid;
    int kernelBuildGrid;
    int kernelApplySPHForces;
    int kernelApplyDrag;
    int kernelUpdateMotion;
    int kernelUpdateRotation;
    int kernelCopyPositions;
    int kernelCopyRotations;

    int selectedParticleID = -1;
    Vector3 dragTargetWorld;
    Vector3[] cpuParticlePositions;
    Quaternion[] cpuParticleRotations;

    private float currentDragDistance;
    private int activeParticleCount = 1; // Class-level variable to track active particles

    // Timer variables for cell splitting
    private float[] cellSplitTimers;
    
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
    
    // List to track pending cell splits
    private List<CellSplitData> pendingSplits = new List<CellSplitData>();

    // Struct to track adhesion between particles 
    private struct ParticleAdhesion
    {
        public int particleA;
        public int particleB;
        public Color connectionColor;
        public bool isFirstGeneration; // Track if this is a first-generation connection
    }
    
    // List to track all adhesion connections
    private List<ParticleAdhesion> adhesionConnections = new List<ParticleAdhesion>();
    
    // Dictionary to track particle parentage for adhesion management
    private Dictionary<int, int> particleParentage = new Dictionary<int, int>();
    
    // Line renderer for visualizing adhesion connections
    private LineRenderer[] adhesionLineRenderers;

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

    // Add member variables to track readback requests
    private bool particleDataReadbackInProgress = false;
    private Particle[] cachedParticleData;
    
    // Add a flag to track if our cached particle data is ready/valid
    private bool cachedParticleDataValid = false;

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
            }
            catch (System.InvalidOperationException ex)
            {
                Debug.LogError($"Error in genome configuration: {ex.Message}");
                enabled = false; // Disable this component to prevent the simulation from running
                return;
            }
            
            CellGenome.OnGenomeChanged += OnGenomeChanged;
        }

        // Initialize all buffers
        InitializeBuffers();
        
        // Initialize particles with genome properties
        InitializeParticles();
        
        // Initialize adhesion line renderers
        InitializeAdhesionLineRenderers();
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
        computeShader.Dispatch(kernelApplySPHForces, threadGroups, 1, 1);

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
        computeShader.Dispatch(kernelCopyRotations, threadGroups, 1, 1);

        uint[] args = new uint[5];
        drawArgsBufferSpheres.GetData(args);
        args[1] = (uint)activeParticleCount; // Use activeParticleCount instead of total count
        drawArgsBufferSpheres.SetData(args);

        sphereMaterial.SetBuffer("particleBuffer", particleBuffer);
        
        Graphics.DrawMeshInstancedIndirect(
            sphereMesh, 0, sphereMaterial,
            new Bounds(Vector3.zero, Vector3.one * spawnRadius * 2f),
            drawArgsBufferSpheres);

        UpdateDragVisualization();
        UpdateAdhesionVisualization();
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

    private void UpdateAdhesionVisualization()
    {
        // Log the number of adhesion connections for debugging
        if (adhesionConnections.Count > 0)
        {
            Debug.Log($"Updating {adhesionConnections.Count} adhesion connections");
        }

        if (!showAdhesionConnections) 
        {
            // Hide all connection lines if visualization is disabled
            if (adhesionLineRenderers != null)
            {
                foreach (var lr in adhesionLineRenderers)
                {
                    if (lr != null)
                    {
                        lr.enabled = false;
                    }
                }
            }
            return;
        }
        
        // Remove invalid connections (e.g., if a particle was removed)
        CleanupInvalidAdhesionConnections();
        
        // Update line positions for all adhesion connections
        for (int i = 0; i < adhesionConnections.Count; i++)
        {
            if (i >= adhesionLineRenderers.Length) break;
            
            var connection = adhesionConnections[i];
            LineRenderer lr = adhesionLineRenderers[i];
            
            // Make sure both particles are within active range
            if (connection.particleA >= activeParticleCount || connection.particleB >= activeParticleCount)
            {
                lr.enabled = false;
                continue;
            }
            
            // Get updated positions from cpuParticlePositions
            Vector3 posA = cpuParticlePositions[connection.particleA];
            Vector3 posB = cpuParticlePositions[connection.particleB];
            
            // Update line positions
            lr.SetPosition(0, posA);
            lr.SetPosition(1, posB);
            
            // Set line color based on connection, but ensure it's highly visible
            Color color = connection.connectionColor;
            // Make sure alpha is high enough
            color.a = 1.0f;
            lr.startColor = color;
            lr.endColor = color;
            
            lr.enabled = true;
        }
        
        // Disable any unused line renderers
        for (int i = adhesionConnections.Count; i < adhesionLineRenderers.Length; i++)
        {
            if (adhesionLineRenderers[i] != null)
            {
                adhesionLineRenderers[i].enabled = false;
            }
        }
    }
    
    private void CleanupInvalidAdhesionConnections()
    {
        // Remove adhesion connections for particles that no longer exist or are beyond activeParticleCount
        adhesionConnections.RemoveAll(c => 
            c.particleA >= activeParticleCount || 
            c.particleB >= activeParticleCount);
    }

    void HandleMouseDrag()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            float closestDist = Mathf.Infinity;
            int closestID = -1;
            
            for (int i = 0; i < cpuParticlePositions.Length; i++)
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
                dragTargetWorld = cpuParticlePositions[selectedParticleID];
                
                currentDragDistance = Vector3.Distance(Camera.main.transform.position, dragTargetWorld);
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

    void OnDestroy()
    {
        // Unsubscribe from genome events
        if (genome != null)
        {
            CellGenome.OnGenomeChanged -= OnGenomeChanged;
        }

        // Release all buffers
        ReleaseBuffers();
    }

    void OnGenomeChanged()
    {
        // Reinitialize particles when genome changes
        if (particleBuffer != null)
        {
            InitializeParticles();
        }
    }

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
        kernelUpdateRotation    = computeShader.FindKernel("UpdateRotation");
        kernelCopyPositions     = computeShader.FindKernel("CopyPositionsToReadbackBuffer");
        kernelCopyRotations     = computeShader.FindKernel("CopyRotationsToReadbackBuffer");

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

        if (circleRenderer != null) Destroy(circleRenderer.gameObject);
        if (lineRenderer != null) Destroy(lineRenderer.gameObject);
        
        if (adhesionLineRenderers != null)
        {
            for (int i = 0; i < adhesionLineRenderers.Length; i++)
            {
                if (adhesionLineRenderers[i] != null)
                {
                    Destroy(adhesionLineRenderers[i].gameObject);
                }
            }
        }
    }

    private void InitializeParticles()
    {
        // Clear any existing adhesion connections when resetting
        adhesionConnections.Clear();
        particleParentage.Clear();
        
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
                    float splitInterval = genome.modes[modeIndex].splitInterval;
                    cellSplitTimers[i] = UnityEngine.Random.Range(0f, splitInterval * 0.8f);
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
        
        for (int i = 0; i < activeParticleCount; i++)
        {
            cellSplitTimers[i] += deltaTime;
        }
        
        const float epsilon = 0.001f;

        for (int i = 0; i < activeParticleCount; i++)
        {
            int modeIndex;
            
            if (cachedParticleDataValid)
            {
                modeIndex = cachedParticleData[i].modeIndex;
            }
            else if (i == 0)
            {
                Particle[] particleData = new Particle[1];
                particleBuffer.GetData(particleData, 0, 0, 1);
                modeIndex = particleData[0].modeIndex;
                
                if (modeIndex < 0 || modeIndex >= genome.modes.Count)
                {
                    modeIndex = GetInitialModeIndex();
                }
            }
            else
            {
                continue;
            }
            
            if (modeIndex >= 0 && modeIndex < genome.modes.Count)
            {
                float splitInterval = genome.modes[modeIndex].splitInterval;
                
                if (cellSplitTimers[i] >= splitInterval - epsilon && allowedSplits > 0)
                {
                    SplitCell(i);
                    cellSplitTimers[i] = 0f;
                    allowedSplits--;
                }
            }
        }
        
        if (cachedParticleDataValid)
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

        Particle[] particleData = new Particle[particleCount];
        particleBuffer.GetData(particleData);

        foreach (var split in pendingSplits)
        {
            if (activeParticleCount + 1 > particleCount)
            {
                Debug.LogWarning("Cannot process split - reached maximum cell count");
                break;
            }

            int childB_Index = activeParticleCount;
            int parentIndex = split.parentIndex;

            // Remove any old adhesion connections for this parent only
            RemoveAdhesionConnectionsForParticle(parentIndex);

            cpuParticlePositions[parentIndex] = split.positionA;
            cpuParticleRotations[parentIndex] = split.rotationA;

            particleData[parentIndex].velocity = split.velocityA;
            particleData[parentIndex].position = split.positionA;
            particleData[parentIndex].rotation = split.rotationA;

            particleData[childB_Index] = particleData[parentIndex];
            cpuParticlePositions[childB_Index] = split.positionB;
            cpuParticleRotations[childB_Index] = split.rotationB;
            particleData[childB_Index].velocity = split.velocityB;
            particleData[childB_Index].position = split.positionB;
            particleData[childB_Index].rotation = split.rotationB;

            GenomeMode parentMode = null;
            if (genome != null && genome.modes.Count > 0)
            {
                int parentModeIndex = particleData[parentIndex].modeIndex;
                if (parentModeIndex >= 0 && parentModeIndex < genome.modes.Count)
                {
                    parentMode = genome.modes[parentModeIndex];
                }

                // Only make sibling adhesion
                if (parentMode != null && parentMode.parentMakeAdhesion)
                {
                    CreateAdhesionConnection(parentIndex, childB_Index, parentMode);
                }
            }

            activeParticleCount++;
            cellSplitTimers[parentIndex] = 0f;
            cellSplitTimers[childB_Index] = 0f;
        }

        particleBuffer.SetData(particleData);
        pendingSplits.Clear();
    }

    private Vector3 GetDirection(float yaw, float pitch)
    {
        return Quaternion.Euler(pitch, yaw, 0f) * Vector3.forward;
    }

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
                
                cachedParticleDataValid = true;
                particleDataReadbackInProgress = false;
            });
        }
        
        AsyncGPUReadback.Request(positionReadbackBuffer, r => 
        {
            if (!r.hasError)
            {
                r.GetData<Vector3>().CopyTo(cpuParticlePositions);
            }
        });
        
        AsyncGPUReadback.Request(rotationReadbackBuffer, r => 
        {
            if (!r.hasError)
            {
                r.GetData<Quaternion>().CopyTo(cpuParticleRotations);
            }
        });
    }

    private void InitializeAdhesionLineRenderers()
    {
        // Check if we need to create new line renderers or if existing ones need adjustment
        if (adhesionLineRenderers == null || adhesionLineRenderers.Length != particleCount)
        {
            // If any existing line renderers exist, destroy them first
            if (adhesionLineRenderers != null)
            {
                for (int i = 0; i < adhesionLineRenderers.Length; i++)
                {
                    if (adhesionLineRenderers[i] != null)
                    {
                        Destroy(adhesionLineRenderers[i].gameObject);
                    }
                }
            }

            // Create new array of line renderers, one per potential adhesion
            adhesionLineRenderers = new LineRenderer[particleCount];
            
            // Create a parent GameObject to organize the line renderers in the hierarchy
            GameObject adhesionLinesParent = new GameObject("AdhesionLines");
            adhesionLinesParent.transform.SetParent(transform);
            
            for (int i = 0; i < particleCount; i++)
            {
                GameObject lineObj = new GameObject($"AdhesionLine_{i}");
                lineObj.transform.SetParent(adhesionLinesParent.transform);
                
                LineRenderer lr = lineObj.AddComponent<LineRenderer>();
                lr.startWidth = adhesionLineWidth;
                lr.endWidth = adhesionLineWidth;
                lr.positionCount = 2;
                
                // Create material with better visibility
                Material lineMaterial = new Material(Shader.Find("Sprites/Default"));
                lineMaterial.renderQueue = 3000; // Ensure it renders on top
                lr.material = lineMaterial;
                lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                lr.receiveShadows = false;
                lr.enabled = false;
                
                adhesionLineRenderers[i] = lr;
            }
        }
        else
        {
            // Update existing line renderers with current width setting
            for (int i = 0; i < adhesionLineRenderers.Length; i++)
            {
                if (adhesionLineRenderers[i] != null)
                {
                    adhesionLineRenderers[i].startWidth = adhesionLineWidth;
                    adhesionLineRenderers[i].endWidth = adhesionLineWidth;
                }
            }
        }
    }

    private void CreateAdhesionConnection(int particleA, int particleB, GenomeMode mode)
    {
        // Always create a sibling adhesion connection when splitting
        ParticleAdhesion adhesion = new ParticleAdhesion
        {
            particleA = particleA,
            particleB = particleB,
            connectionColor = mode.modeColor,
            isFirstGeneration = false
        };
        adhesionConnections.Add(adhesion);
        Debug.Log($"Created adhesion connection between particles {particleA} and {particleB}");
    }

    private void RemoveAdhesionConnectionsForParticle(int particleIndex)
    {
        // Remove all adhesion connections involving the specified particle
        adhesionConnections.RemoveAll(c => c.particleA == particleIndex || c.particleB == particleIndex);
    }
}