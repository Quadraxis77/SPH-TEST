using UnityEngine;
using System.Collections.Generic;

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
    [Range(1f, 20f)] public float globalSplitTimerMin = 5f;
    [Range(0.1f, 5f)] public float spawnOverlapOffset = 0.5f;
    [Range(0.1f, 10f)] public float splitVelocityMagnitude = 0.5f;
    public int maxCells = 64;

    [Header("Genome Settings")]
    public CellGenome genome;

    [Header("Simulation Assets")]
    public ComputeShader computeShader;
    public Material sphereMaterial;
    public Mesh sphereMesh;

    [Header("Drag Visualization Settings")]
    public Color dragCircleColor = Color.green;
    public float dragCircleRadius = 1.0f;

    private LineRenderer circleRenderer;
    private LineRenderer lineRenderer;

    ComputeBuffer particleBuffer, dragInputBuffer, drawArgsBufferSpheres;
    ComputeBuffer positionReadbackBuffer, rotationReadbackBuffer;
    ComputeBuffer gridHeads, gridNext, gridParticleIndices;
    ComputeBuffer torqueAccumBuffer;
    ComputeBuffer genomeModesBuffer; // Add persistent genomeModesBuffer

    const int GRID_DIM = 32;
    const int GRID_TOTAL = GRID_DIM * GRID_DIM * GRID_DIM;

    int kernelInitParticles;
    int kernelClearGrid;
    int kernelBuildGrid;
    int kernelApplySPHForces;
    int kernelApplyAdhesionForces;
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

    void Start()
    {
        Application.targetFrameRate = 144;
        
        // Subscribe to genome changes
        if (genome != null)
        {
            CellGenome.OnGenomeChanged += OnGenomeChanged;
        }

        // Initialize all buffers
        InitializeBuffers();
        
        // Initialize particles with genome properties
        InitializeParticles();
    }

    void Update()
    {
        float dt = Time.deltaTime;
        int threadGroups = Mathf.CeilToInt(particleCount / 64f);

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
        
        // No need to set adhesion parameters here as they come from the genome

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

        computeShader.SetBuffer(kernelApplyAdhesionForces, "particleBuffer", particleBuffer);
        computeShader.SetBuffer(kernelApplyAdhesionForces, "gridHeads", gridHeads);
        computeShader.SetBuffer(kernelApplyAdhesionForces, "gridNext", gridNext);
        computeShader.SetBuffer(kernelApplyAdhesionForces, "gridParticleIndices", gridParticleIndices);
        computeShader.SetBuffer(kernelApplyAdhesionForces, "torqueAccumBuffer", torqueAccumBuffer);
        // Only set the genomeModesBuffer if it's not null
        if (genomeModesBuffer != null)
        {
            computeShader.SetBuffer(kernelApplyAdhesionForces, "genomeModesBuffer", genomeModesBuffer);
        }
        computeShader.Dispatch(kernelApplyAdhesionForces, threadGroups, 1, 1);

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

        positionReadbackBuffer.GetData(cpuParticlePositions);
        rotationReadbackBuffer.GetData(cpuParticleRotations);

        uint[] args = new uint[5];
        drawArgsBufferSpheres.GetData(args);
        args[1] = (uint)activeParticleCount; // Use activeParticleCount instead of total count
        drawArgsBufferSpheres.SetData(args);

        sphereMaterial.SetBuffer("particleBuffer", particleBuffer);
        // Only pass the genome mode buffer to the material if it's not null
        if (genomeModesBuffer != null)
        {
            // Pass the genome mode buffer and default mode to the material for coloring
            sphereMaterial.SetBuffer("genomeModesBuffer", genomeModesBuffer);
            sphereMaterial.SetInt("defaultGenomeMode", GetInitialModeIndex());
        }
        
        Graphics.DrawMeshInstancedIndirect(
            sphereMesh, 0, sphereMaterial,
            new Bounds(Vector3.zero, Vector3.one * spawnRadius * 2f),
            drawArgsBufferSpheres);

        UpdateDragVisualization();
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

    void HandleMouseDrag()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            float closestDist = Mathf.Infinity;
            int closestID = -1;
            Vector3 hitPoint = Vector3.zero;

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
                        hitPoint = ray.origin + ray.direction * t;
                    }
                }
            }

            if (closestID != -1)
            {
                selectedParticleID = closestID;
                dragTargetWorld = hitPoint;
                
                // Store the initial distance from camera to dragged particle
                currentDragDistance = Vector3.Distance(Camera.main.transform.position, dragTargetWorld);
            }
        }

        if (Input.GetMouseButton(0) && selectedParticleID != -1)
        {
            // Project the mouse position to world space at the initially captured distance
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            
            // Set the drag target at the same distance from the camera as when initially clicked
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
        kernelApplyAdhesionForces = computeShader.FindKernel("ApplyAdhesionForces");
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
        genomeModesBuffer?.Release();

        if (circleRenderer != null) Destroy(circleRenderer.gameObject);
        if (lineRenderer != null) Destroy(lineRenderer.gameObject);
    }

    private void InitializeParticles()
    {
        computeShader.SetFloat("spawnRadius", spawnRadius);
        computeShader.SetFloat("minRadius", minRadius);
        computeShader.SetFloat("maxRadius", maxRadius);
        computeShader.SetFloat("torqueFactor", torqueFactor);
        computeShader.SetFloat("torqueDamping", torqueDamping);
        computeShader.SetFloat("boundaryFriction", boundaryFriction);
        computeShader.SetFloat("rollingContactRadiusMultiplier", rollingContactRadiusMultiplier);
        computeShader.SetFloat("density", density);
        computeShader.SetFloat("repulsionStrength", repulsionStrength);
        
        // Release any existing genomeModesBuffer
        if (genomeModesBuffer != null)
        {
            genomeModesBuffer.Release();
            genomeModesBuffer = null;
        }
        
        // Instead of initializing all 10,000 particles, we'll only initialize what's needed
        // based on the genome configuration
        int initialParticleCount = 1; // Start with just a single particle
        
        // Pass genome mode information to the compute shader
        if (genome != null && genome.modes.Count > 0)
        {
            // Get initial genome mode for default adhesion settings
            GenomeMode initialMode = null;
            int initialModeIndex = GetInitialModeIndex();
            
            if (initialModeIndex >= 0 && initialModeIndex < genome.modes.Count)
            {
                initialMode = genome.modes[initialModeIndex];
                
                // Set adhesion parameters from the initial genome mode
                computeShader.SetFloat("adhesionRestLength", initialMode.adhesionRestLength);
                computeShader.SetFloat("adhesionSpringStiffness", initialMode.adhesionSpringStiffness);
                computeShader.SetFloat("adhesionSpringDamping", initialMode.adhesionSpringDamping);
            }
            
            // Create a struct containing just the adhesion properties we need
            GenomeAdhesionData[] genomeData = new GenomeAdhesionData[genome.modes.Count];
            
            for (int i = 0; i < genome.modes.Count; i++)
            {
                GenomeMode mode = genome.modes[i];
                genomeData[i] = new GenomeAdhesionData
                {
                    parentMakeAdhesion = mode.parentMakeAdhesion ? 1 : 0,
                    childA_KeepAdhesion = mode.childA_KeepAdhesion ? 1 : 0,
                    childB_KeepAdhesion = mode.childB_KeepAdhesion ? 1 : 0,
                    adhesionRestLength = mode.adhesionRestLength,
                    adhesionSpringStiffness = mode.adhesionSpringStiffness,
                    adhesionSpringDamping = mode.adhesionSpringDamping,
                    // Pack RGB color into a uint for easy transfer
                    colorPacked = PackColor(mode.modeColor),
                    // Add orientation constraint parameters
                    orientConstraintStrength = mode.orientationConstraintStrength,
                    maxAngleDeviation = mode.maxAllowedAngleDeviation
                };
            }
            
            // Create and set the buffer as a persistent buffer
            genomeModesBuffer = new ComputeBuffer(genome.modes.Count, 
                                                 sizeof(int) * 3 + 
                                                 sizeof(float) * 5 + // 3 adhesion params + 2 orientation params
                                                 sizeof(uint));
            genomeModesBuffer.SetData(genomeData);
            
            // Set the genome parameters
            computeShader.SetInt("genomeModesCount", genome.modes.Count);
            computeShader.SetInt("defaultGenomeMode", initialModeIndex);
            computeShader.SetInt("activeParticleCount", initialParticleCount);
            
            // Set the buffer for all kernels that might need it
            SetGenomeModesBufferForAllKernels();
        }
        else
        {
            // No genome data, set count to 0
            computeShader.SetInt("genomeModesCount", 0);
            computeShader.SetInt("activeParticleCount", initialParticleCount);
            
            // Create a minimal default buffer to satisfy the shader
            CreateDefaultGenomeModesBuffer();
        }

        computeShader.SetBuffer(kernelInitParticles, "particleBuffer", particleBuffer);
        computeShader.SetBuffer(kernelInitParticles, "torqueAccumBuffer", torqueAccumBuffer);
        computeShader.Dispatch(kernelInitParticles, Mathf.CeilToInt(particleCount / 64f), 1, 1);
    }

    // Helper method to create a minimal default genome buffer
    private void CreateDefaultGenomeModesBuffer()
    {
        if (genomeModesBuffer == null)
        {
            // Create a minimal buffer with one default entry
            GenomeAdhesionData[] defaultData = new GenomeAdhesionData[1];
            defaultData[0] = new GenomeAdhesionData
            {
                parentMakeAdhesion = 0,
                childA_KeepAdhesion = 0,
                childB_KeepAdhesion = 0,
                adhesionRestLength = 3.0f,
                adhesionSpringStiffness = 100.0f,
                adhesionSpringDamping = 5.0f,
                colorPacked = PackColor(Color.green),
                orientConstraintStrength = 0.5f,
                maxAngleDeviation = 45.0f
            };

            genomeModesBuffer = new ComputeBuffer(1, 
                                sizeof(int) * 3 + 
                                sizeof(float) * 5 + 
                                sizeof(uint));
            genomeModesBuffer.SetData(defaultData);
        }
        
        // Set the buffer for all kernels that might need it
        SetGenomeModesBufferForAllKernels();
    }

    // Helper method to set the genome buffer for all kernels
    private void SetGenomeModesBufferForAllKernels()
    {
        if (genomeModesBuffer != null)
        {
            // Setting the buffer for all kernels to prevent errors
            // This ensures any shader that might use the buffer has it available
            int kernelCount = computeShader.FindKernel("InitParticles");
            for (int i = 0; i <= kernelCount; i++)
            {
                try
                {
                    computeShader.SetBuffer(i, "genomeModesBuffer", genomeModesBuffer);
                }
                catch (System.Exception)
                {
                    // Ignore errors for kernels that don't use this buffer
                }
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

    // Helper struct to pass just the needed genome data to the GPU
    private struct GenomeAdhesionData
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

    // Helper method to pack a Color into a uint
    private uint PackColor(Color color)
    {
        uint r = (uint)(color.r * 255.0f);
        uint g = (uint)(color.g * 255.0f);
        uint b = (uint)(color.b * 255.0f);
        return (r << 16) | (g << 8) | b;
    }

    private void UpdateCellDivisionTimers(float deltaTime)
    {
        // Initialize timer array if needed
        if (cellSplitTimers == null || cellSplitTimers.Length < activeParticleCount)
        {
            // Create or resize timer array
            float[] newTimers = new float[particleCount];
            if (cellSplitTimers != null)
            {
                // Copy existing timers if we're resizing
                System.Array.Copy(cellSplitTimers, newTimers, cellSplitTimers.Length);
            }
            cellSplitTimers = newTimers;
        }
        
        // Process pending splits first
        if (pendingSplits.Count > 0)
        {
            ProcessPendingSplits();
        }
        
        // Check if we can add more cells
        int allowedSplits = maxCells - activeParticleCount;
        if (allowedSplits <= 0) return;
        
        // Update timers and check for splits
        for (int i = 0; i < activeParticleCount; i++)
        {
            cellSplitTimers[i] += deltaTime;
            
            // Check if it's time to split and we still have room
            if (cellSplitTimers[i] >= globalSplitTimerMin && allowedSplits > 0)
            {
                SplitCell(i);
                cellSplitTimers[i] = 0f; // Reset timer
                allowedSplits--;
            }
        }
    }
    
    // Handle particle cell division
    private void SplitCell(int parentIndex)
    {
        if (genome == null || genome.modes.Count == 0 || parentIndex >= activeParticleCount)
            return;
            
        // Get the parent cell's position and orientation
        Vector3 parentPos = cpuParticlePositions[parentIndex];
        Quaternion parentRot = cpuParticleRotations[parentIndex];
        
        // Get the parent's actual mode index by reading from the particle buffer
        Particle[] particleData = new Particle[1];
        // Fix: Properly read a single particle from the specific index in the buffer
        particleBuffer.GetData(particleData, 0, parentIndex, 1);
        int parentModeIndex = particleData[0].modeIndex;
        
        // Ensure parent mode index is valid
        if (parentModeIndex < 0 || parentModeIndex >= genome.modes.Count) {
            parentModeIndex = GetInitialModeIndex();
        }
        
        // Get the genome mode for this cell using the parent's actual mode
        GenomeMode mode = genome.modes[parentModeIndex];
        
        // Get the child mode indices
        int childAModeIndex = mode.childAModeIndex;
        if (childAModeIndex < 0 || childAModeIndex >= genome.modes.Count)
            childAModeIndex = parentModeIndex; // Fallback to parent mode
            
        int childBModeIndex = mode.childBModeIndex;
        if (childBModeIndex < 0 || childBModeIndex >= genome.modes.Count)
            childBModeIndex = parentModeIndex; // Fallback to parent mode
        
        // Calculate split direction
        Vector3 forward = parentRot * Vector3.forward;
        Vector3 up = parentRot * Vector3.up;
        Vector3 right = parentRot * Vector3.right;
        
        // Get split direction from genome (in local space)
        Vector3 splitDirLocal = GetDirection(mode.parentSplitYaw, mode.parentSplitPitch);
        
        // Convert to world space
        Vector3 splitDirWorld = right * splitDirLocal.x + up * splitDirLocal.y + forward * splitDirLocal.z;
        
        // Child A and B positions
        Vector3 posA = parentPos + splitDirWorld * spawnOverlapOffset;
        Vector3 posB = parentPos - splitDirWorld * spawnOverlapOffset;
        
        // Child orientations
        Vector3 childADirLocal = GetDirection(mode.childA_OrientationYaw, mode.childA_OrientationPitch);
        Vector3 childADirWorld = right * childADirLocal.x + up * childADirLocal.y + forward * childADirLocal.z;
        Quaternion rotA = Quaternion.LookRotation(childADirWorld, up);
        
        Vector3 childBDirLocal = GetDirection(mode.childB_OrientationYaw, mode.childB_OrientationPitch);
        Vector3 childBDirWorld = right * childBDirLocal.x + up * childBDirLocal.y + forward * childBDirLocal.z;
        Quaternion rotB = Quaternion.LookRotation(childBDirWorld, up);
        
        // Get parent velocity from physics simulation
        // In a full implementation we'd read the particle buffer, for now we'll assume zero
        Vector3 parentVelocity = Vector3.zero;
        
        // Create velocities for children
        Vector3 velA = parentVelocity + splitDirWorld * splitVelocityMagnitude;
        Vector3 velB = parentVelocity - splitDirWorld * splitVelocityMagnitude;
        
        // Create the split data
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
        
        // Add to pending splits
        pendingSplits.Add(splitData);
    }
    
    // Process all pending cell splits
    private void ProcessPendingSplits()
    {
        if (pendingSplits.Count == 0)
            return;
        
        // Create a copy of the particle data that we can modify
        Particle[] particleData = new Particle[particleCount];
        particleBuffer.GetData(particleData);
        
        foreach (var split in pendingSplits)
        {
            if (activeParticleCount + 1 > maxCells) // Only +1 because we reuse the parent
            {
                Debug.LogWarning("Cannot process split - reached maximum cell count");
                break;
            }
            
            // Child B will be the new particle
            int childB_Index = activeParticleCount;
            
            // Update parent particle (becomes Child A)
            cpuParticlePositions[split.parentIndex] = split.positionA;
            cpuParticleRotations[split.parentIndex] = split.rotationA;
            
            // Set child A velocity and genome mode
            particleData[split.parentIndex].velocity = (Vector3)split.velocityA;
            
            // Copy parent data for Child B (with modifications)
            particleData[childB_Index] = particleData[split.parentIndex];
            
            // Set child B position, rotation, velocity
            cpuParticlePositions[childB_Index] = split.positionB;
            cpuParticleRotations[childB_Index] = split.rotationB;
            particleData[childB_Index].velocity = (Vector3)split.velocityB;
            particleData[childB_Index].position = (Vector3)split.positionB;
            
            // Set proper genome flags based on mode
            if (genome != null && genome.modes.Count > 0)
            {
                int childAModeIndex = split.childAModeIndex;
                int childBModeIndex = split.childBModeIndex;
                
                if (childAModeIndex >= 0 && childAModeIndex < genome.modes.Count)
                {
                    GenomeMode modeA = genome.modes[childAModeIndex];
                    uint flagsA = 0;
                    
                    if (modeA.parentMakeAdhesion)
                        flagsA |= 2; // GENOME_MAKES_ADHESION
                    
                    flagsA |= 1; // GENOME_HAS_ADHESION (all cells can receive adhesion)
                    
                    if (modeA.childA_KeepAdhesion)
                        flagsA |= 4; // GENOME_CHILD_A_KEEP_ADHESION
                        
                    if (modeA.childB_KeepAdhesion)
                        flagsA |= 8; // GENOME_CHILD_B_KEEP_ADHESION
                        
                    particleData[split.parentIndex].genomeFlags = flagsA;
                    particleData[split.parentIndex].orientConstraintStr = modeA.orientationConstraintStrength;
                    particleData[split.parentIndex].modeIndex = childAModeIndex; // Set mode index for Child A
                }
                
                if (childBModeIndex >= 0 && childBModeIndex < genome.modes.Count)
                {
                    GenomeMode modeB = genome.modes[childBModeIndex];
                    uint flagsB = 0;
                    
                    if (modeB.parentMakeAdhesion)
                        flagsB |= 2; // GENOME_MAKES_ADHESION
                    
                    flagsB |= 1; // GENOME_HAS_ADHESION (all cells can receive adhesion)
                    
                    if (modeB.childA_KeepAdhesion)
                        flagsB |= 4; // GENOME_CHILD_A_KEEP_ADHESION
                        
                    if (modeB.childB_KeepAdhesion)
                        flagsB |= 8; // GENOME_CHILD_B_KEEP_ADHESION
                        
                    particleData[childB_Index].genomeFlags = flagsB;
                    particleData[childB_Index].orientConstraintStr = modeB.orientationConstraintStrength;
                    particleData[childB_Index].modeIndex = childBModeIndex; // Set mode index for Child B
                }
            }
            
            // Increment active particle count
            activeParticleCount++;
            
            // Reset timers for both cells
            cellSplitTimers[split.parentIndex] = 0f;
            cellSplitTimers[childB_Index] = 0f;
        }
        
        // Write the updated particle data back to the GPU
        particleBuffer.SetData(particleData);
        
        // Clear pending splits
        pendingSplits.Clear();
    }
    
    // Utility function to convert pitch and yaw to a direction vector
    private Vector3 GetDirection(float yaw, float pitch)
    {
        return Quaternion.Euler(pitch, yaw, 0f) * Vector3.forward;
    }
}