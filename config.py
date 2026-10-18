"""全局配置: 路径 / 命名 / 频率 / 视觉标定 / 动作原语参数."""

from __future__ import annotations

import os
from pathlib import Path

# ---- 路径 ----
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR  # 项目根
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
EPISODES_DIR = DATA_DIR / "episodes"
ACT_DATA_DIR = DATA_DIR / "act_v1"

BASE_XML = MODELS_DIR / "base_chassis.xml"
SCENE_XML = MODELS_DIR / "scene_pickball.xml"
GENERATED_XML = MODELS_DIR / "mobile_manip.generated.xml"
ARM_XML = MODELS_DIR / "reference_arm.xml"

# ---- 模型命名 ----
ARM_PREFIX = "arm_"
JOINT_BASE = "base_ref"
JOINT_WHEEL_L = "wheel_left_joint"
JOINT_WHEEL_R = "wheel_right_joint"
ACT_WHEEL_L = "wheel_left_vel"
ACT_WHEEL_R = "wheel_right_vel"
BODY_BASE = "base"

JOINT_SHOULDER = f"{ARM_PREFIX}shoulder_joint"
JOINT_ELBOW = f"{ARM_PREFIX}elbow_joint"
JOINT_FINGER = f"{ARM_PREFIX}finger_slide"
ACT_SHOULDER = f"{ARM_PREFIX}shoulder_servo"
ACT_ELBOW = f"{ARM_PREFIX}elbow_servo"
ACT_FINGER = f"{ARM_PREFIX}finger_servo"
SITE_TCP = f"{ARM_PREFIX}gripper_tcp"
BODY_GRIPPER = f"{ARM_PREFIX}gripper_tip"
BODY_ARM_BASE = f"{ARM_PREFIX}arm_base"

# 指滑动 ctrl: 0 = 全开, 0.015 = 轻贴球面
FINGER_OPEN_CTRL = 0.0
FINGER_CLOSE_CTRL = 0.05    # 满行程闭合: 左指面到+0.027, 与固定右指(-0.077)夹住0.12m球; 行程不足时正对球差2mm永远无接触
FINGER_CLOSED_THRESHOLD = 0.008   # 仅识别闭爪命令意图，不代表手指实际闭合
GRIPPER_CLOSE_SECONDS = 1.2       # 单次闭爪时间上限；确认夹持且观测稳定后可提前结束
GRASP_DIST = 0.15                 # 球心距 TCP 此距离内才做接触检测(预筛)
GRASP_CONTACT_TOL_M = 0.003        # 指垫内表面接触位置容差
GRASP_NORMAL_COS_MIN = 0.7        # 接触法向须主要沿指垫内表面法向
GRASP_OPENING_MIN = 0.05
GRASP_OPENING_MAX = 0.45
GRASP_STABLE_SECONDS = 0.1        # 同一个球连续双侧接触的确认窗口
GRASP_OPENING_TOL = 0.02          # 窗口内实测开度极差（对应 1 mm 行程）
GRASP_LOSS_SECONDS = 0.1          # 持续失去有效夹持后解除辅助吸附

# ---- 相机 (车头前向, 唯一的"眼睛") ----
CAM_FRONT = "front_cam"
CAM_FRONT_RES = (320, 240)        # 感知分辨率 (W, H)
CAM_FRONT_FOVY = 75.0             # 垂直视场角 (度, 与 XML 一致)
CAM_HEIGHT = 0.26                 # 相机离地高度 (m)
CAM_PITCH_DEG = 30.0              # 光轴俯角 (度)
CAM_FORWARD_OFFSET = 0.16         # 相机相对底盘中心的前移量 (m)

# 记录用相机 (不给大脑, 只给人看回放)
CAM_OVERHEAD = "overhead"
CAM_SIDE = "side"
OBS_CAMS = [CAM_FRONT]            # 大脑观测相机
REC_CAMS = [CAM_FRONT, CAM_OVERHEAD]

# ---- 频率 / 步长 ----
SIM_DT = 0.002
CTRL_HZ = 10
CTRL_DT = 1.0 / CTRL_HZ

# 正常结束决策后继续运行的仿真秒数，再评定结果；继续采集，不调用模型。
# 保持臂/爪目标并停止底盘；按控制周期向上取整，0 禁用。异常和关窗跳过。
EPISODE_SETTLE_SECONDS = 3.0

# ---- 动作原语 (低层时间盒控制) ----
PRIM_V = 0.30          # forward/back 速度 (m/s)
PRIM_W = 1.0           # turn_left/right 角速度 (rad/s)
PRIM_MAX_S = 3.0       # 单次命令最长秒数
OBSERVE_SECONDS = 0.1  # 默认观察上限；至少推进一拍，稳定后可提前结束
OBSERVE_MAX_SECONDS = 3.0
BASE_SETTLE_SECONDS = 0.2
ARM_SETTLE_SECONDS = 0.3
GRIPPER_OPEN_SECONDS = 0.8
ARM_SETTLE_GOAL_TOL_RAD = 0.03  # 提前结束等待的关节误差范数；原末尾失败检查仍保留
GRIPPER_OPEN_GOAL_TOL = 0.02    # 张开目标容差（归一化开度）
ARM_NUDGE_MAX = 0.5    # shoulder/elbow 单次微调最大弧度

# ---- 相邻控制帧的近似静止判定（仅控制器等待使用，不修改归档图片） ----
STABILITY_BLOCK_PIXELS = 3
STABILITY_IMAGE_EPS = 5.0          # 每块每通道平均绝对差，RGB 0–255
STABILITY_IMAGE_SIMILARITY = 0.998
STABILITY_LOCAL_PIXELS = 24        # 每个重叠局部区域最多允许 10% 的块变化
STABILITY_LOCAL_CHANGED_RATIO = 0.10
STABILITY_STATE_EPS = (0.001, 0.001, 0.002, 0.002, 0.005)  # rad, rad, 开度, m/s, rad/s
STABILITY_BASE_SPEED_EPS = (0.002, 0.005)  # 还须速度本身接近零，不能把匀速当静止

# ---- 臂预设姿态 [肩, 肘] ----
# 同一正肘分支，避免收放时绕过高位；以下为真实伺服稳定后的近似 TCP 位置。
# 前伸量相对底盘中心，高度相对地面；球心随实际夹取位置变化，不等于 TCP。
ARM_STOW = [0.0, 2.2]       # 空爪折叠，车头视野畅通；TCP≈前0.32m、高0.44m
ARM_REACH = [1.1414, 1.4639]  # 地面抓取；比旧姿态低约5mm；TCP≈前0.545m、高0.094m，指尖离地≈0.052m
ARM_CARRY = [0.95, 1.10]    # 抬高持球/运输；TCP≈前0.67m、高0.31m，球顶部可能部分出画
ARM_DROP = [0.5161, 2.0053]  # 短臂靠箱投放；TCP≈前0.42m、高0.30m，保留大部分球像；看图确认球下方为箱口内部

# ---- 任务场景 ----
BALL_NAMES = ["ball_0", "ball_1", "ball_2", "ball_3"]
BALL_R = 0.06
BALL_SPAWN_X = (0.45, 1.75)
BALL_SPAWN_Y = (-1.30, 1.30)
BIN_POS = (1.35, 0.0)
BIN_HALF_INT = (0.17, 0.17)
BIN_WALL_TOP = 0.16
BASE_SPAWN = (-0.9, 0.0, 0.0)

# ---- 视觉检测参数 ----
BALL_HUE = (55, 100)    # 网球黄绿色相范围 (度)
BALL_SAT = 0.35
BALL_VAL = 0.25
BALL_MIN_AREA = 25      # 像素
# 收纳箱=洋红色 (perception._bin_mask 颜色比判别)
BIN_SAT = 0.30
BIN_VAL = 0.20
BIN_MIN_AREA = 150

# ---- 随机化 ----
RAND_PHYS = 0.2

# ---- Brain / LLM ----
MAX_DECISIONS = 32
LLM_TIMEOUT_S = 180.0   # 大 max_tokens 下思考链长, 单请求可能超 60s
LLM_MAX_TOKENS = 65536
# VISTA 自行读取 .env/进程环境；凭据不进入源码或日志。
LOCAL_LLM_FILE = ROOT / ".env"


def ensure_dirs() -> None:
    for d in (DATA_DIR, EPISODES_DIR):
        os.makedirs(d, exist_ok=True)
