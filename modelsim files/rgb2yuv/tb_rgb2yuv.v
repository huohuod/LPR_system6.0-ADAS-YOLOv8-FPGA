`timescale 1ns / 1ps

module tb_rgb2yuv;

// 时钟
reg clk;

// 输入信号
reg [7:0] r_in;
reg [7:0] g_in;
reg [7:0] b_in;
reg vs_in;
reg hs_in;
reg de_in;

// 增强控制（全部关闭）
reg [7:0] video_enhance_lightdown_num;
reg       video_enhance_lightdown_sw;
reg [7:0] video_enhance_darkup_num;
reg       video_enhance_darkup_sw;

// 输出信号
wire [7:0] y_out;
wire [7:0] u_out;
wire [7:0] v_out;
wire vs_out;
wire hs_out;
wire de_out;

// 例化被测试模块
rgb2yuv u_rgb2yuv (
    .clk                          (clk),
    .r_in                         (r_in),
    .g_in                         (g_in),
    .b_in                         (b_in),
    .vs_in                        (vs_in),
    .hs_in                        (hs_in),
    .de_in                        (de_in),
    
    .y_out                        (y_out),
    .u_out                        (u_out),
    .v_out                        (v_out),
    .vs_out                       (vs_out),
    .hs_out                       (hs_out),
    .de_out                       (de_out),
    
    .video_enhance_lightdown_num  (8'd0),
    .video_enhance_lightdown_sw   (1'b0),
    .video_enhance_darkup_num     (8'd0),
    .video_enhance_darkup_sw      (1'b0)
);

// 时钟 50MHz
always #10 clk = ~clk;

// 激励
initial begin
    clk = 0;
    r_in = 0;
    g_in = 0;
    b_in = 0;
    vs_in = 0;
    hs_in = 0;
    de_in = 0;
    
    video_enhance_lightdown_num = 0;
    video_enhance_lightdown_sw  = 0;
    video_enhance_darkup_num    = 0;
    video_enhance_darkup_sw     = 0;
    
    #50;
    vs_in = 1;
    hs_in = 1;
    de_in = 1;
    
    // 给一组纯红色
    #20;
    r_in = 255;
    g_in = 0;
    b_in = 0;
    
    // 给一组纯绿色
    #40;
    r_in = 0;
    g_in = 255;
    b_in = 0;
    
    // 给一组纯蓝色
    #40;
    r_in = 0;
    g_in = 0;
    b_in = 255;
    
    // 给一组白色
    #40;
    r_in = 255;
    g_in = 255;
    b_in = 255;
    
    #200;
    $stop;
end

endmodule