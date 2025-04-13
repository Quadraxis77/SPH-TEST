Shader "Custom/InstancedParticles_URP"
{
    Properties
    {
        _Color ("Base Color", Color) = (1,1,1,1)
        _ShowDot ("Show Forward Dot", Float) = 1.0
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode"="UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            // Struct layout must exactly match compute buffer (80 bytes total)
            struct Particle
            {
                float3 position;
                float radius;

                float3 velocity;
                float mass;

                float3 angularVelocity;
                float momentOfInertia;

                float drag;
                float repulsionStrength;
                float2 padding0;

                float4 rotation;
            };

            StructuredBuffer<Particle> particleBuffer;

            float4 _Color;
            float _ShowDot;

            struct Attributes
            {
                float3 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                uint instanceID   : SV_InstanceID;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 normalWS    : TEXCOORD0;
                float3 worldPos    : TEXCOORD1;
                float3 forwardDir  : TEXCOORD2;
            };

            // Rotate a vector by quaternion
            float3 RotateVector(float3 v, float4 q)
            {
                return v + 2.0f * cross(q.xyz, cross(q.xyz, v) + q.w * v);
            }

            Varyings vert (Attributes input)
            {
                Varyings output;

                Particle p = particleBuffer[input.instanceID];

                // Rotate + scale mesh vertex
                float3 localPos = input.positionOS * p.radius;
                float3 rotatedPos = RotateVector(localPos, p.rotation);
                float3 worldPos = p.position + rotatedPos;

                // Rotate normal
                float3 rotatedNormal = RotateVector(input.normalOS, p.rotation);
                float3 worldNormal = normalize(mul((float3x3)unity_ObjectToWorld, rotatedNormal));

                output.positionHCS = TransformWorldToHClip(worldPos);
                output.normalWS = worldNormal;
                output.worldPos = worldPos;

                // Forward Z+ axis transformed by quaternion
                output.forwardDir = normalize(RotateVector(float3(0, 0, 1), p.rotation));

                return output;
            }

            half4 frag (Varyings input) : SV_Target
            {
                Light light = GetMainLight();
                float3 lightDir = normalize(light.direction);
                float NdotL = saturate(dot(input.normalWS, lightDir));
                float3 baseColor = _Color.rgb * NdotL + 0.1;

                // Optional red highlight if normal faces forward direction
                float NdotF = dot(normalize(input.normalWS), normalize(input.forwardDir));
                float highlight = (_ShowDot > 0.5) ? smoothstep(0.98, 1.0, NdotF) : 0.0;
                float3 redDot = float3(1, 0, 0) * highlight;

                float3 finalColor = baseColor + redDot;
                return float4(finalColor, 1.0);
            }

            ENDHLSL
        }
    }

    FallBack "Diffuse"
}
