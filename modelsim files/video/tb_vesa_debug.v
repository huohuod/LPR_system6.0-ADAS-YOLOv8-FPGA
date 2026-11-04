`timescale 1ns / 1ps

module tb_vesa_debug;

// 输入信号
reg         pix_clk;
reg         rstn;
reg         vs;
reg         de;

// 输出信号
wire [15:0] vesa_data;

// 例化待测模块
vesa_debug #(
    .PIX_WIGHT(16)
)
u_vesa_debug (
    .pix_clk  (pix_clk),
    .rstn     (rstn),
    .vs       (vs),
    .de       (de),
    .vesa_data(vesa_data)
);

// 时钟 50MHz
always #10 pix_clk = ~pix_clk;

// 激励信号（最简单）
initial begin
    pix_clk = 0;
    rstn    = 0;
    vs      = 0;
    de      = 0;
    
    #50;
    rstn = 1;   // 释放复位
    
    #50;
    vs = 1;     // 场同步
    #50;
    vs = 0;
    
    #50;
    de = 1;     // 数据有效开始
    #200;
    de = 0;     // 数据有效结束
    
    #200;
    $stop;
end

endmodule