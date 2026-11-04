`timescale 1ns / 1ps

module tb_ov5640_power_on_delay;

// 输入
reg         clk_50M;
reg         reset_n;

// 输出
wire        camera1_rstn;
wire        camera2_rstn;
wire        camera_pwnd;
wire        initial_en;

// 例化待测模块
ov5640_power_on_delay u_dut (
    .clk_50M        (clk_50M),
    .reset_n        (reset_n),
    .camera1_rstn   (camera1_rstn),
    .camera2_rstn   (camera2_rstn),
    .camera_pwnd    (camera_pwnd),
    .initial_en     (initial_en)
);

// 50MHz 时钟
always #10 clk_50M = ~clk_50M;

// 复位激励
initial begin
    clk_50M = 0;
    reset_n = 0;
    
    #100;
    reset_n = 1;  // 释放复位
    
    #2000000;     // 跑足够长时间看完整上电时序
    $stop;
end

endmodule