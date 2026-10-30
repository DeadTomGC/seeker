from machine import Pin, PWM

max_throws = 80.0
output_scaler = 1.0
servo_pwms = [PWM(Pin(8)),PWM(Pin(9)),PWM(Pin(12)),PWM(Pin(13))]
for servo_pwm in servo_pwms:
    servo_pwm.freq(50)  # Standard servo frequency

def mix_controls(roll: float, pitch: float, yaw: float) -> list[float]:
    """
    Mix roll, pitch, and yaw commands into 4 control surface outputs for a
    symmetric "+" configuration.

    Surface layout (viewed from behind, looking forward):
        0 = top
        1 = starboard
        2 = lower
        3 = port

    Inputs and outputs are in the range -100..100.

    Sign convention (positive command -> positive deflection of a surface
    means the trailing edge deflects in the positive rotation direction
    about that surface's hinge axis, viewed per the right-hand rule):
        - Roll:  all four surfaces deflect in the same rotational sense
        - Pitch: acts through the horizontal surfaces (1 and 3), which
                 deflect in opposite directions
        - Yaw:   acts through the vertical surfaces (0 and 2), which
                 deflect in opposite directions

    Returns:
        [out0, out1, out2, out3], each in -100..100
    """
    global max_throws
    global output_scaler
    # Clamp inputs to valid range
    roll = max(-100.0, min(100.0, roll))
    pitch = max(-100.0, min(100.0, pitch))
    yaw = max(-100.0, min(100.0, yaw))

    # Mixing matrix: each row is [roll, pitch, yaw] contribution to a surface
    mix = [
        # roll, pitch, yaw
        [1.0,   0.0,   1.0],   # 0: top
        [1.0,   1.0,   0.0],   # 1: starboard
        [1.0,   0.0,  -1.0],   # 2: lower
        [1.0,  -1.0,   0.0],   # 3: port
    ]

    # Sum contributions (any single input can reach 100% on its surfaces)
    outputs = [
        (r * roll + p * pitch + y * yaw)*output_scaler
        for r, p, y in mix
    ]

    # Renormalize if any output exceeds 100%
    peak = max(abs(o) for o in outputs)
    if peak > max_throws:
        scale = max_throws / peak
        outputs = [o * scale for o in outputs]

    return outputs

def control_servos(servo_pos:list[float]):
    global servo_pwms
    for i,servo_pwm in enumerate(servo_pwms):
        
        duty = max(min(int(5000 + (servo_pos[i] * (3400) / 100)),8400),1600)
        servo_pwm.duty_u16(duty)

# --- ESC / primary motor ------------------------------------------------
# Standard 50 Hz servo-style throttle signal, 1000..2000 us.
esc_pwm = PWM(Pin(15))
esc_pwm.freq(50)
ESC_IDLE_DUTY = 3277   # ~1000 us
ESC_MAX_DUTY = 6553    # ~2000 us

def control_throttle(throttle_pct: float, armed: bool):
    """Drive the ESC.  throttle_pct 0..100, armed gates the output.
    Unarmed output is a 1000 us idle pulse (kept for ESC keep-alive)."""
    global esc_pwm
    throttle_pct = max(0.0, min(100.0, throttle_pct))
    if armed and throttle_pct > 0.0:
        duty = int(ESC_IDLE_DUTY + (ESC_MAX_DUTY - ESC_IDLE_DUTY) * throttle_pct / 100.0)
    else:
        duty = ESC_IDLE_DUTY
    esc_pwm.duty_u16(duty)

p = 0.1
i = 0.01
d = 0.01
integral = [0,0,0]
old_gyro = [0,0,0]
integral_max = 2000
def update_pid(gyro,pitch,yaw,roll=0.0)->list[float]:
    global integral,old_gyro,integral_max,p,i,d
    pitch_target_dps = 1.0*pitch
    yaw_target_dps = 1.0*yaw
    roll_target_dps = 1.0*roll
    # x is pitch
    # y is yaw
    # z is roll
    pitch_dps = gyro[0]
    yaw_dps = gyro[1]
    roll_dps = gyro[2]
    dist = [pitch_target_dps - pitch_dps, yaw_target_dps - yaw_dps, roll_target_dps - roll_dps]
    deriv = [0,0,0]
    output = [0,0,0]
    for n in range(0,3):
        integral[n] = max(0-integral_max,min(integral_max, integral[n] + dist[n]))
        deriv[n] = gyro[n] - old_gyro[n]
        output[n] = p*dist[n] + i*integral[n] + d*deriv[n]
        old_gyro[n] = gyro[n]
    return output
    
       
    

if __name__ == "__main__":
    print(update_pid((1,1,1),0,0))
    print(mix_controls(100, 0, 0))     # [100, 100, 100, 100]
    print(mix_controls(0, 100, 0))     # [0, 100, 0, -100]
    print(mix_controls(0, 0, 100))     # [100, 0, -100, 0]
    print(mix_controls(100, 100, 100)) # renormalized
    print(control_throttle(50, True))
