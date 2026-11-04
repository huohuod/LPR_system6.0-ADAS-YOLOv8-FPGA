`timescale 1ps/1ps

module tb_AXI_FULL_M;

// 时钟与复位
reg                            M_AXI_ACLK;
reg                            M_AXI_ARESETN;
reg                            DDR_INIT_DONE;

// 写地址通道
wire [27:0]                    M_AXI_AWADDR;
wire                           M_AXI_AWVALID;
reg                            M_AXI_AWREADY;

// 写数据通道
reg                            M_AXI_WLAST;
reg                            M_AXI_WREADY;

// 读地址通道
wire [27:0]                    M_AXI_ARADDR;
wire                           M_AXI_ARVALID;
reg                            M_AXI_ARREADY;

// 读数据通道
reg                            M_AXI_RLAST;
reg                            M_AXI_RVALID;

// 视频信号
reg                            vs_in;
reg                            vs_out;

// FIFO 信号
reg [8:0]                      wfifo_rd_water_level;
wire                           wfifo_rd_req;
wire                           wfifo_pre_rd_req;
reg [8:0]                      rfifo_wr_water_level;
wire                           rfifo_wr_req;
wire                           r_fram_done;

// 地址配置
reg [19:0]                     wr_addr_min;
reg [19:0]                     wr_addr_max;

// 输出
wire                           r_wr_rst;
wire                           r_rd_rst;
wire [1:0]                     w_fifo_state;
wire [1:0]                     r_fifo_state;
wire [19:0]                    wr_addr_cnt;

// 例化待测模块
AXI_FULL_M #(
    .VIDEO_LENGTH        (1920),
    .VIDEO_HIGTH         (1080),
    .PIXEL_WIDTH         (32),
    .CTRL_ADDR_WIDTH     (28),
    .DQ_WIDTH            (32),
    .M_AXI_BRUST_LEN     (8),
    .VIDEO_BASE_ADDR     (0)
)
u_AXI_FULL_M (
    .DDR_INIT_DONE       (DDR_INIT_DONE),
    .M_AXI_ACLK          (M_AXI_ACLK),
    .M_AXI_ARESETN       (M_AXI_ARESETN),

    .M_AXI_AWADDR        (M_AXI_AWADDR),
    .M_AXI_AWVALID       (M_AXI_AWVALID),
    .M_AXI_AWREADY       (M_AXI_AWREADY),

    .M_AXI_WLAST         (M_AXI_WLAST),
    .M_AXI_WREADY        (M_AXI_WREADY),

    .M_AXI_ARADDR        (M_AXI_ARADDR),
    .M_AXI_ARVALID       (M_AXI_ARVALID),
    .M_AXI_ARREADY       (M_AXI_ARREADY),

    .M_AXI_RLAST         (M_AXI_RLAST),
    .M_AXI_RVALID        (M_AXI_RVALID),

    .vs_in               (vs_in),
    .vs_out              (vs_out),

    .wfifo_rd_water_level(wfifo_rd_water_level),
    .wfifo_rd_req        (wfifo_rd_req),
    .wfifo_pre_rd_req    (wfifo_pre_rd_req),

    .rfifo_wr_water_level(rfifo_wr_water_level),
    .rfifo_wr_req        (rfifo_wr_req),
    .r_fram_done         (r_fram_done),

    .wr_addr_min         (wr_addr_min),
    .wr_addr_max         (wr_addr_max),

    .r_wr_rst            (r_wr_rst),
    .r_rd_rst            (r_rd_rst),
    .w_fifo_state        (w_fifo_state),
    .r_fifo_state        (r_fifo_state),
    .wr_addr_cnt         (wr_addr_cnt)
);

// 时钟 100MHz
always #5000 M_AXI_ACLK = ~M_AXI_ACLK;

// 复位
initial begin
    M_AXI_ACLK = 0;
    M_AXI_ARESETN = 0;
    DDR_INIT_DONE = 0;
    vs_in = 1;
    vs_out = 1;
    M_AXI_AWREADY = 0;
    M_AXI_WREADY = 0;
    M_AXI_WLAST = 0;
    M_AXI_ARREADY = 0;
    M_AXI_RVALID = 0;
    M_AXI_RLAST = 0;
    wfifo_rd_water_level = 0;
    rfifo_wr_water_level = 512;
    wr_addr_min = 0;
    wr_addr_max = 10000;
    #20000;
    M_AXI_ARESETN = 1;
    #30000;
    DDR_INIT_DONE = 1;
    #50000;
    wfifo_rd_water_level = 20;
end

// 简单激励：给一个帧同步
initial begin
    #100000;
    vs_in = 0;
    #10000;
    vs_in = 1;
end

initial begin
    #200000;
    vs_out = 0;
    #10000;
    vs_out = 1;
end

// 简单应答激励
always @(posedge M_AXI_ACLK) begin
    M_AXI_AWREADY <= M_AXI_AWVALID;
    M_AXI_ARREADY <= M_AXI_ARVALID;
    M_AXI_WREADY <= 1;
    M_AXI_RVALID <= 1;
end

// 仿真结束
initial begin
    #500000;
    $stop;
end

endmodule