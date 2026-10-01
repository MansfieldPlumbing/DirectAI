using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace DirectAI;

public enum GpuVendor
{
    Unknown,
    Nvidia,
    Amd,
    Intel,
    Qualcomm,
    Microsoft
}

public class GpuDeviceInfo
{
    public int DirectMlIndex { get; set; }
    public string Description { get; set; }
    public GpuVendor Vendor { get; set; }
    public uint VendorId { get; set; }
    public uint DeviceId { get; set; }
    public long DedicatedVideoMemoryBytes { get; set; }
    public long SharedSystemMemoryBytes { get; set; }
    public long AdapterLuid { get; set; }
    public bool IsHardware { get; set; }

    public double DedicatedMemoryGB => Math.Round(DedicatedVideoMemoryBytes / (1024.0 * 1024.0 * 1024.0), 2);
    public double SharedMemoryGB => Math.Round(SharedSystemMemoryBytes / (1024.0 * 1024.0 * 1024.0), 2);

    public override string ToString() =>
        $"[Device {DirectMlIndex}] {Description} | {Vendor} (VendorId=0x{VendorId:X4}, DeviceId=0x{DeviceId:X4}) | VRAM: {DedicatedMemoryGB} GB | Shared: {SharedMemoryGB} GB";
}

public class DeviceTopology
{
    public List<GpuDeviceInfo> Devices { get; set; } = new();

    public int TextEncoderDeviceId { get; set; } = 0;
    public int UnetDeviceId { get; set; } = 0;
    public int VaeDecoderDeviceId { get; set; } = 0;
    public int ControlNetDeviceId { get; set; } = 0;

    public bool IsMultiGpu => Devices.Count > 1;

    public void AutoAssignStages()
    {
        if (Devices.Count == 0) return;

        if (Devices.Count == 1)
        {
            TextEncoderDeviceId = Devices[0].DirectMlIndex;
            UnetDeviceId = Devices[0].DirectMlIndex;
            VaeDecoderDeviceId = Devices[0].DirectMlIndex;
            ControlNetDeviceId = Devices[0].DirectMlIndex;
            return;
        }

        // Multi-GPU / Multi-Die assignment:
        // Filter compute devices (prefer hardware adapters with Dedicated VRAM)
        var computeDevices = Devices.FindAll(d => d.IsHardware && d.DedicatedVideoMemoryBytes > 0);
        if (computeDevices.Count == 0) computeDevices = Devices;

        // Stage 0: Text Encoder (Lightweight: ~500 MB - 1 GB)
        TextEncoderDeviceId = computeDevices[0].DirectMlIndex;

        // Stage 1: UNet (Heavy compute & resident memory: ~1.7 - 3 GB)
        UnetDeviceId = computeDevices.Count > 1 ? computeDevices[1].DirectMlIndex : computeDevices[0].DirectMlIndex;

        // Stage 2: VAE Decoder (~150 MB - 300 MB)
        VaeDecoderDeviceId = computeDevices.Count > 2 ? computeDevices[2].DirectMlIndex : computeDevices[0].DirectMlIndex;

        // Stage 3: ControlNet (~1.5 GB)
        ControlNetDeviceId = computeDevices.Count > 3 ? computeDevices[3].DirectMlIndex : computeDevices[1 % computeDevices.Count].DirectMlIndex;
    }
}

[System.Runtime.Versioning.SupportedOSPlatform("windows")]
public static class DeviceManager
{
    [DllImport("dxgi.dll", ExactSpelling = true)]
    private static extern int CreateDXGIFactory1(ref Guid riid, out IntPtr ppFactory);

    private static readonly Guid IID_IDXGIFactory1 = new("770aae78-f26f-4dba-a829-253c83d1b387");

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct DXGI_ADAPTER_DESC1
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string Description;
        public uint VendorId;
        public uint DeviceId;
        public uint SubSysId;
        public uint Revision;
        public UIntPtr DedicatedVideoMemory;
        public UIntPtr DedicatedSystemMemory;
        public UIntPtr SharedSystemMemory;
        public long AdapterLuid;
        public uint Flags;
    }

    [ComImport]
    [Guid("770aae78-f26f-4dba-a829-253c83d1b387")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIFactory1
    {
        [PreserveSig] int SetPrivateData(ref Guid Name, uint DataSize, IntPtr pData);
        [PreserveSig] int SetPrivateDataInterface(ref Guid Name, IntPtr pUnknown);
        [PreserveSig] int GetPrivateData(ref Guid Name, ref uint pDataSize, IntPtr pData);
        [PreserveSig] int GetParent(ref Guid riid, out IntPtr ppParent);
        [PreserveSig] int EnumAdapters(uint Adapter, out IntPtr ppAdapter);
        [PreserveSig] int MakeWindowAssociation(IntPtr WindowHandle, uint Flags);
        [PreserveSig] int GetWindowAssociation(out IntPtr pWindowHandle);
        [PreserveSig] int CreateSwapChain(IntPtr pDevice, IntPtr pDesc, out IntPtr ppSwapChain);
        [PreserveSig] int CreateSoftwareAdapter(IntPtr Module, out IntPtr ppAdapter);
        [PreserveSig] int EnumAdapters1(uint Adapter, out IntPtr ppAdapter);
        [PreserveSig] int IsCurrent();
    }

    [ComImport]
    [Guid("29038f61-3839-4626-91fd-086879011a05")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIAdapter1
    {
        [PreserveSig] int SetPrivateData(ref Guid Name, uint DataSize, IntPtr pData);
        [PreserveSig] int SetPrivateDataInterface(ref Guid Name, IntPtr pUnknown);
        [PreserveSig] int GetPrivateData(ref Guid Name, ref uint pDataSize, IntPtr pData);
        [PreserveSig] int GetParent(ref Guid riid, out IntPtr ppParent);
        [PreserveSig] int EnumOutputs(uint Output, out IntPtr ppOutput);
        [PreserveSig] int GetDesc(IntPtr pDesc);
        [PreserveSig] int CheckInterfaceSupport(ref Guid InterfaceName, out long pUMDVersion);
        [PreserveSig] int GetDesc1(out DXGI_ADAPTER_DESC1 pDesc);
    }

    public static DeviceTopology DiscoverTopology()
    {
        var topology = new DeviceTopology();
        var riid = IID_IDXGIFactory1;
        int hr = CreateDXGIFactory1(ref riid, out IntPtr factoryPtr);
        if (hr != 0 || factoryPtr == IntPtr.Zero)
        {
            Console.WriteLine($"[DeviceManager] CreateDXGIFactory1 failed (HRESULT 0x{hr:X8}). Falling back to single device 0.");
            topology.Devices.Add(new GpuDeviceInfo { DirectMlIndex = 0, Description = "Default DirectML Device", Vendor = GpuVendor.Unknown });
            topology.AutoAssignStages();
            return topology;
        }

        var factory = (IDXGIFactory1)Marshal.GetObjectForIUnknown(factoryPtr);
        uint adapterIndex = 0;

        while (factory.EnumAdapters1(adapterIndex, out IntPtr adapterPtr) == 0)
        {
            var adapter = (IDXGIAdapter1)Marshal.GetObjectForIUnknown(adapterPtr);
            if (adapter.GetDesc1(out DXGI_ADAPTER_DESC1 desc) == 0)
            {
                bool isSoftware = (desc.Flags & 2) != 0; // DXGI_ADAPTER_FLAG_SOFTWARE
                var vendor = desc.VendorId switch
                {
                    0x10DE => GpuVendor.Nvidia,
                    0x1002 => GpuVendor.Amd,
                    0x8086 => GpuVendor.Intel,
                    0x5143 => GpuVendor.Qualcomm,
                    0x1414 => GpuVendor.Microsoft,
                    _ => GpuVendor.Unknown
                };

                topology.Devices.Add(new GpuDeviceInfo
                {
                    DirectMlIndex = (int)adapterIndex,
                    Description = desc.Description,
                    Vendor = vendor,
                    VendorId = desc.VendorId,
                    DeviceId = desc.DeviceId,
                    DedicatedVideoMemoryBytes = (long)desc.DedicatedVideoMemory.ToUInt64(),
                    SharedSystemMemoryBytes = (long)desc.SharedSystemMemory.ToUInt64(),
                    AdapterLuid = desc.AdapterLuid,
                    IsHardware = !isSoftware
                });
            }

            Marshal.Release(adapterPtr);
            adapterIndex++;
        }

        Marshal.Release(factoryPtr);
        topology.AutoAssignStages();
        return topology;
    }
}
