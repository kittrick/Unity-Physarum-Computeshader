using System;
using System.Collections.Generic;
using EasyButtons;
using UnityEngine;

public class Physarum : MonoBehaviour
{
    // ------------------------------
    // Primary CCA Parameters
    // ------------------------------
    [Header("Trail Agents Settings")]
    [Range(64, 1000000)]
    public int agentsCount = 64;

    private ComputeBuffer agentsBuffer;

    [Range(0, 1)]
    public float trailDecayFactor = .9f;
    [Range(1,10)]
    public int diffusionRange = 1;
    [Range(1, 360)]
    public int sensorCount = 3;
    [Range(0, 1080)]
    public float sensorRange = 1;
    [Range(0, 360)]
    public float sensorAngle = 45;

    [Header("Physics")]
    [Range(0, 100)]
    public float mass = 1;
    [Range(0, 100)]
    public float drag = 0;
    [Range(0, 100)]
    public float speed = 1;
    [Range(0, 100)]
    public float sensorForce = 1;

    [Header("Mouse Input")]
    [Range(0, 100)]
    public int brushSize = 10;
    public GameObject interactivePlane;
    protected Vector2 hitXY;

    // ------------------------------
    // Global Parameters
    // ------------------------------
    [Header("Setup")]
    public Camera cam;

    [Range(8, 2048)]
    public int rez = 512;

    [Range(0, 50)]
    public int stepsPerFrame = 1;

    [Range(1, 50)]
    public int stepMod = 1;

    public ComputeShader cs;
    public Material outMat;
    
    private RenderTexture outTex;
    private RenderTexture readTex;
    private RenderTexture writeTex;
    private RenderTexture debugTex;
    
    private int moveAgentsKernel;
    private int writeTrailsKernel;
    private int renderKernel;
    private int diffuseTextureKernel;
    
    protected List<ComputeBuffer> buffers;
    protected List<RenderTexture> textures;

    protected int stepn = -1;

    
    // ------------------------------
    // Reset
    // ------------------------------
    [Button]
    private void Reset()
    {
        Release();
        moveAgentsKernel = cs.FindKernel("MoveAgentsKernel");
        renderKernel = cs.FindKernel("RenderKernel");
        writeTrailsKernel = cs.FindKernel("WriteTrailsKernel");
        diffuseTextureKernel = cs.FindKernel("DiffuseTextureKernel");

        readTex = CreateTexture(rez, FilterMode.Point);
        writeTex = CreateTexture(rez, FilterMode.Point);
        outTex = CreateTexture(rez, FilterMode.Point);
        debugTex = CreateTexture(rez, FilterMode.Point);
        
        agentsBuffer = new ComputeBuffer(agentsCount, sizeof(float) * 4);
        buffers.Add(agentsBuffer);
        
        GPUResetKernel();
        Render();
    }
    
    private void GPUResetKernel()
    {
        int kernel;
        
        cs.SetInt("rez", rez);
        cs.SetInt("time", Time.frameCount);

        kernel = cs.FindKernel("ResetTextureKernel");
        cs.SetTexture(kernel, "writeTex", writeTex);
        cs.Dispatch(kernel, rez / 32, rez / 32, 1);
        
        cs.SetTexture(kernel, "writeTex", readTex);
        cs.Dispatch(kernel, rez / 32, rez / 32, 1);

        kernel = cs.FindKernel("ResetAgentsKernel");
        cs.SetBuffer(kernel, "agentsBuffer", agentsBuffer);
        cs.Dispatch(kernel, agentsCount / 64, 1,1);
    }
    
    // ------------------------------
    // Start
    // ------------------------------
    private void Start()
    {
        Reset();    
    }

    // ------------------------------
    // Update
    // ------------------------------
    private void Update()
    {
        if(Time.frameCount % stepMod == 0)
        {
            for (int i = 0; i < stepsPerFrame; i++)
            {
                Step();
            }
        }
    }

    // ------------------------------
    // Step
    // ------------------------------
    [Button]
    public void Step()
    {
        HandleInput();
        
        stepn += 1;
        cs.SetInt("time", Time.frameCount);
        cs.SetInt("stepn", stepn);
        cs.SetInt("brushSize", brushSize);
        cs.SetVector("hitXY", hitXY);
        
        GPUMoveAgentsKernel();

        if (stepn % 2 == 1)
        {
            GPUDiffuseTextureKernel();
            GPUWriteTrailsKernel();
            SwapTex();
        }

        Render();
    }

    void HandleInput()
    {
        
        if (!Input.GetMouseButton(0))
        {
            hitXY.x = hitXY.y = 0;
            return;
        }

        RaycastHit hit;
        if (Physics.Raycast(cam.ScreenPointToRay(Input.mousePosition), out hit, Mathf.Infinity))
        {
            if (hit.transform != interactivePlane.transform)
            {
                return;
            }
            hitXY = hit.textureCoord * rez;
        }
    }

    private void GPUDiffuseTextureKernel()
    {
        cs.SetTexture(diffuseTextureKernel, "readTex", readTex);
        cs.SetTexture(diffuseTextureKernel, "writeTex", writeTex);
        cs.SetFloat("trailDecayFactor", trailDecayFactor);
        cs.SetInt("diffusionRange", diffusionRange);
        cs.SetInt("sensorCount", sensorCount);
        cs.SetFloat("sensorRange", sensorRange);
        cs.SetFloat("sensorAngle", sensorAngle);
        cs.SetFloat("mass", mass);
        cs.SetFloat("drag", drag);
        cs.SetFloat("speed", speed);
        cs.SetFloat("sensorForce", sensorForce);
        cs.Dispatch(diffuseTextureKernel, rez / 32, rez / 32, 1);
    }

    private void GPUMoveAgentsKernel()
    {
        cs.SetBuffer(moveAgentsKernel, "agentsBuffer", agentsBuffer);
        cs.SetTexture(moveAgentsKernel, "readTex", readTex);
        cs.SetTexture(moveAgentsKernel, "debugTex", debugTex);
        
        cs.Dispatch(moveAgentsKernel, agentsCount / 64, 1, 1);
    }
    
    private void GPUWriteTrailsKernel()
    {
        cs.SetBuffer(writeTrailsKernel, "agentsBuffer", agentsBuffer);
        
        cs.SetTexture(writeTrailsKernel, "writeTex", writeTex);
        
        cs.Dispatch(writeTrailsKernel, agentsCount / 64, 1, 1);
    }

    private void SwapTex()
    {
        RenderTexture tmp = readTex;
        readTex = writeTex;
        writeTex = tmp;
    }

    public void Render()
    {
        GPURenderKernel();

        outMat.SetTexture("_BaseMap", outTex);
        outMat.SetTexture("_EmissionMap", outTex);
        if (!Application.isPlaying)
        {
            UnityEditor.SceneView.RepaintAll();
        }
    }
    
    private void GPURenderKernel()
    {
       cs.SetTexture(renderKernel, "readTex", readTex);
       cs.SetTexture(renderKernel, "outTex", outTex);
       cs.SetTexture(renderKernel, "debugTex", debugTex);
       cs.SetVector("hitXY", hitXY);
       cs.Dispatch(renderKernel, rez / 32, rez / 32, 1);
    }

    // ------------------------------
    // Helper Functions
    // ------------------------------
    public void Release()
    {
        if (buffers != null)
        {
            foreach (ComputeBuffer buffer in buffers)
            {
                if (buffer != null)
                {
                    buffer.Release();
                    buffer.Dispose();
                }
            }
        }
        
        buffers = new List<ComputeBuffer>();

        if (textures != null)
        {
            foreach (RenderTexture tex in textures)
            {
                if (tex != null)
                {
                    tex.Release();
                }
            }
        }
        
        textures = new List<RenderTexture>();
    }

    private void OnDestroy()
    {
        Release();
    }
    
    private void OnEnable()
    {
        Release();
    }
    
    private void OnDisable()
    {
        Release();
    }

    private void OnApplicationQuit()
    {
        Release();
    }

    protected RenderTexture CreateTexture(int r, FilterMode filterMode)
    {
        RenderTexture texture = new RenderTexture(r, r, 1, RenderTextureFormat.ARGBFloat);

        texture.name = "out";
        texture.enableRandomWrite = true;
        texture.dimension = UnityEngine.Rendering.TextureDimension.Tex2D;
        texture.volumeDepth = 1;
        texture.filterMode = filterMode;
        texture.wrapMode = TextureWrapMode.Repeat;
        texture.autoGenerateMips = false;
        texture.useMipMap = false;
        texture.Create();
        textures.Add(texture);
        
        return texture;
    }
}
