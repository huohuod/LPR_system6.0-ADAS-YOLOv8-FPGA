`timescale 1ns / 1ps

module tb_yuv2rgb;

// 时钟
reg clk;

// 输入信号
reg [7:0] y_in;
reg [7:0] u_in;
reg [7:0] v_in;
reg vs_in;
reg hs_in;
reg de_in;

// 输出信号
wire [7:0] r_out;
wire [7:0] g_out;
wire [7:0] b_out;
wire vs_out;
wire hs_out;
wire de_out;

// 例化被测试模块
yuv2rgb u_yuv2rgb (
    .clk    (clk),
    .y_in   (y_in),
    .u_in   (u_in),
    .v_in   (v_in),
    .vs_in  (vs_in),
    .hs_in  (hs_in),
    .de_in  (de_in),
    
    .r_out  (r_out),
    .g_out  (g_out),
    .b_out  (b_out),
    .vs_out (vs_out),
    .hs_out (hs_out),
    .de_out (de_out)
);

// 50MHz 时钟
always #10 clk = ~clk;

// 激励信号（最简单的测试）
initial begin
    clk = 0;
    y_in = 0;
    u_in = 0;
    v_in = 0;
    vs_in = 0;
    hs_in = 0;
    de_in = 0;
    
    #50;
    // 给有效信号
    vs_in = 1;
    hs_in = 1;
    de_in = 1;
    
    // 给一组 YUV 数据，模块会自动转成 RGB
    #20;
    y_in = 8'd100;
    u_in = 8'd128;
    v_in = 8'd128;
    
    // 再给一组
    #40;
    y_in = 8'd200;
    u_in = 8'd150;
    v_in = 8'd100;
    
    #200;
    $stop;
end

endmodule