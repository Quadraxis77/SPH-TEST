using UnityEngine;

/// <summary>
/// Manages free camera movement and orbit within the simulation.
/// </summary>
public class CameraFly : MonoBehaviour
{
    #region Singleton Implementation
    public static CameraFly Instance { get; private set; }

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(this.gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(this.gameObject);
    }
    #endregion

    #region Public Settings
    [Header("Camera Movement Settings")]
    public float moveSpeed = 10f;
    public float sprintMultiplier = 2.0f;

    [Header("Rotation Settings")]
    public float lookSensitivity = 2f;
    public bool invertLook = false;
    private float yaw = 0f; 
    private float pitch = 0f;

    [Header("Zoom Settings")]
    public float zoomSpeed = 200f;
    public float minDistance = 5f;
    public float maxDistance = 100f;
    public float zoomSmoothing = 5f;
    public float zoomIncrement = 20f;

    [Header("Orbit Mode")]
    public bool orbitMode = false;
    public Transform orbitTarget;
    public float orbitDistance = 10f;
    private float orbitYaw = 0f;
    private float orbitPitch = 0f;

    private Vector3 moveDirection;
    private float currentZoom;
    private float targetZoom;
    private bool isDragging = false;
    #endregion

    #region Initialization
    private void Start()
    {
        Vector3 startEuler = transform.eulerAngles;
        yaw = (startEuler.y > 180f) ? startEuler.y - 360f : startEuler.y;
        pitch = (startEuler.x > 180f) ? startEuler.x - 360f : startEuler.x;
        pitch = Mathf.Clamp(pitch, -80f, 80f);
        
        currentZoom = orbitDistance;
        targetZoom = currentZoom;
    }
    #endregion

    #region Update Loop
    private void Update()
    {
        HandleZoom();
        HandleCameraMode();

        if (orbitMode && orbitTarget != null)
        {
            OrbitAroundTarget();
        }
        else
        {
            HandleMovement();
            HandleRotation();
        }
    }
    #endregion

    #region Camera Control Methods

    private void HandleMovement()
    {
        float speed = moveSpeed * (Input.GetKey(KeyCode.LeftShift) ? sprintMultiplier : 1f);
        moveDirection = Vector3.zero;

        if (Input.GetKey(KeyCode.W)) moveDirection += transform.forward;
        if (Input.GetKey(KeyCode.S)) moveDirection -= transform.forward;
        if (Input.GetKey(KeyCode.A)) moveDirection -= transform.right;
        if (Input.GetKey(KeyCode.D)) moveDirection += transform.right;
        if (Input.GetKey(KeyCode.Q)) moveDirection -= Vector3.up;
        if (Input.GetKey(KeyCode.E)) moveDirection += Vector3.up;

        transform.position += moveDirection * speed * Time.deltaTime;
    }

    private void HandleRotation()
    {
        if (Input.GetMouseButtonDown(1)) isDragging = true;
        if (Input.GetMouseButtonUp(1)) isDragging = false;

        if (isDragging)
        {
            float mouseX = Input.GetAxis("Mouse X") * lookSensitivity;
            float mouseY = Input.GetAxis("Mouse Y") * lookSensitivity * (invertLook ? -1 : 1);

            yaw += mouseX;
            pitch = Mathf.Clamp(pitch - mouseY, -80f, 80f);

            transform.rotation = Quaternion.Euler(pitch, yaw, 0);
        }
    }

    private void HandleZoom()
    {
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (scroll != 0f)
        {
            targetZoom = Mathf.Clamp(targetZoom - scroll * zoomIncrement, minDistance, maxDistance);
        }
        currentZoom = Mathf.Lerp(currentZoom, targetZoom, Time.deltaTime * zoomSmoothing);
        transform.position += transform.forward * scroll * zoomSpeed * Time.deltaTime;
    }

    private void OrbitAroundTarget()
    {
        if (orbitTarget == null) return;

        Vector3 direction = new Vector3(0, 0, -orbitDistance);
        Quaternion rotation = Quaternion.Euler(orbitPitch, orbitYaw, 0);
        transform.position = orbitTarget.position + rotation * direction;
        transform.LookAt(orbitTarget);
    }

    private void HandleCameraMode()
    {
        if (Input.GetKeyDown(KeyCode.O))
        {
            orbitMode = !orbitMode;
        }
    }

    #endregion

    #region Focus Function (Fix for CS1061)

    /// <summary>
    /// Moves the camera to focus on a given cell.
    /// </summary>
    /// <param name="cellTransform">The cell to focus on.</param>
    public void FocusOnCell(Transform cellTransform)
    {
        if (cellTransform == null)
        {
            Debug.LogWarning("FocusOnCell called with a null transform!");
            return;
        }

        orbitTarget = cellTransform;
        orbitMode = true;

        // Set camera position slightly behind the cell
        transform.position = cellTransform.position - (cellTransform.forward * orbitDistance);
        transform.LookAt(cellTransform);
    }

    #endregion
}
