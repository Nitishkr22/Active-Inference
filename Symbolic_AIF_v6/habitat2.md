 ▐▛███▜▌   Claude Code v2.1.186
▝▜█████▛▘  Sonnet 4.6 · Claude Pro
  ▘▘ ▝▝    ~/…/AIF_code/Active-Inference/Symbolic_AIF_v6/habitat_adapter


● I have a complete picture of everything. Here's the design before touching
  code:

  Two new EFE terms:
  1. Wall penalty: vectorized line-segment distance from each imagined final
  pose to every wall slot → Gaussian penalty. Prevents planner from driving into
  walls.
  2. Doorway routing: when wall slots block the direct path to goal → redirect
  goal_pos to the best doorway slot (minimise detour cost: dist_agent→door +
  dist_door→goal). After agent passes through, revert to original goal.

  New planner state: _nav_goal (true end goal), _via_door flag, _door_target.
  reached_goal() always checks _nav_goal, not the effective door waypoint.

  Let me implement it:

● Update(~/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/model_v6/config.p
        y)
  ⎿  Added 9 lines, removed 1 line
      159      # Frontier exploration: penalise candidates whose final po
           sition is too
      160      # close to already-visited positions (drives the agent tow
           ard novel areas).
      161      w_frontier_penalty: float = 1.5   # penalty weight
      162 -    frontier_radius: float = 1.0     # metres — "already explo
          -red" radius                                                   
      162 +    frontier_radius: float = 1.0      # metres — "already expl
          +ored" radius                                                  
      163  
      164 +    # Stage 2 — Wall-awareness and doorway routing            
      165 +    # Wall repulsion: penalise sequences whose path crosses ne
          +ar a wall slot.                                               
      166 +    w_wall_penalty:   float = 5.0     # EFE penalty weight    
      167 +    wall_margin_m:    float = 0.8     # Gaussian sigma for wal
          +l influence (metres)                                          
      168 +    # Doorway routing: redirect goal through a doorway slot wh
          +en wall blocks path.                                          
      169 +    door_block_margin_m: float = 0.70 # wall within this dist 
          +of path-to-goal = blocked                                     
      170 +    door_pass_radius_m:  float = 0.80 # dist to doorway slot c
          +ounts as "passed through"                                     
      171  
      172 +                                                              
      173  @dataclass
      174  class ModelV6Config:
      175      perception:  PerceptionConfig  = field(default_factory=Per
           ceptionConfig)
  ⎿  Read ../model_v6/config.py (189 lines)
  ⎿  Read ../../../../../.claude/projects/-home-nitish-Documents-AIF-code/memory
     /project_aif_v6_explore_nav.md (170 lines)
  ⎿  Read ../model_v6/structural_detector.py (267 lines)
  ⎿  Referenced file aif_explore_nav_demo.py
  ⎿  Read ../model_v6/__init__.py (24 lines)

● Good, config is updated. Now let me read the current planner in full before
  modifying it.

  Read 1 file (ctrl+o to expand)

● Now I have the full current planner. Let me write the Stage 2 version:

● Write(~/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/planner/efe_planne
       r_v4.py)
  ⎿  Added 244 lines, removed 64 lines
       4  Replaces V3's discrete 9×9 grid with continuous pose Gaussian.
       5  Uses WorldModelV6.imagine_rollout() for batched GPU imagination
          .
       6  
       7 -EFE = Risk + Ambiguity - InfoGain + ActionCost                 
       7 +EFE = Risk + Ambiguity - InfoGain + ActionCost + WallPenalty   
       8  
       9  Risk       : expected negative log-preference at the goal.
      10               P*(s) ∝ exp(-||pose[:2] - goal||² / (2 * r²))
     ...
      18  
      19  ActionCost : small penalty to prefer efficient actions.
      20  
      21 +WallPenalty: Gaussian repulsion from wall slots (Stage 2).     
      22 +             Routes the agent through doorway slots when direct
         + path blocked.                                                 
      23 +                                                               
      24  Sequences  : 3^K (forward=0, turn_left=1, turn_right=2) — same 
          count as V5.
      25               K=5, N=243.
      26  """
     ...
      35  
      36  import sys, os
      37  sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."
          ))
      35 -from model_v6 import BeliefState, WorldModelV6                 
      38 +from model_v6 import BeliefState, WorldModelV6, WALL_ID, DOORWA
         +Y_ID                                                           
      39  from model_v6.config import EFEConfig
      40  
      41  
      42 +# ── Module-level helpers ─────────────────────────────────────
         +──────────────────                                             
      43 +                                                               
      44  def _round_heading(theta: float, turn_step: float) -> float:
      45      """Round theta to the nearest multiple of turn_step, normal
          ised to [-π, π]."""
      46      n = round(theta / turn_step)
     ...
       51      return math.atan2(math.sin(theta), math.cos(theta))
       52  
       53  
       54 +def _extract_structural_slots(                                
       55 +    belief:   BeliefState,                                    
       56 +    slot_cfg,                                                 
       57 +) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:   
       58 +    """                                                       
       59 +    Extract wall and doorway slot positions (XY) from the beli
          +ef map.                                                       
       60 +                                                              
       61 +    Returns:                                                  
       62 +        wall_xy : [W, 2] float32  or  None if no wall slots   
       63 +        door_xy : [D, 2] float32  or  None if no doorway slots
       64 +    """                                                       
       65 +    occupied = belief.slot_conf_logit > slot_cfg.conf_logit_em
          +pty_threshold                                                 
       66 +    if not occupied.any():                                    
       67 +        return None, None                                     
       68 +                                                              
       69 +    cls = belief.slot_class_logits[occupied].argmax(dim=-1)   
          +# [M]                                                         
       70 +    pos = belief.slot_pos_mu[occupied]                        
          + # [M, 3]                                                     
       71 +                                                              
       72 +    wall_mask = cls == WALL_ID                                
       73 +    door_mask = cls == DOORWAY_ID                             
       74 +                                                              
       75 +    wall_xy = pos[wall_mask, :2] if wall_mask.any() else None 
       76 +    door_xy = pos[door_mask, :2] if door_mask.any() else None 
       77 +    return wall_xy, door_xy                                   
       78 +                                                              
       79 +                                                              
       80 +def _is_path_blocked(                                         
       81 +    curr_xy:  torch.Tensor,   # [2]                           
       82 +    goal_xy:  torch.Tensor,   # [2]                           
       83 +    wall_xy:  torch.Tensor,   # [W, 2]                        
       84 +    margin:   float,                                          
       85 +) -> bool:                                                    
       86 +    """                                                       
       87 +    Return True if any wall slot lies within `margin` metres o
          +f the line segment                                            
       88 +    curr_xy → goal_xy (only the interior portion, t ∈ [0.1, 0.
          +9]).                                                          
       89 +    """                                                       
       90 +    if wall_xy is None or wall_xy.numel() == 0:               
       91 +        return False                                          
       92 +    W      = wall_xy.shape[0]                                 
       93 +    device = wall_xy.device                                   
       94 +    A = curr_xy.to(device)                                    
       95 +    B = goal_xy.to(device)                                    
       96 +    BA = (B - A).unsqueeze(0).expand(W, 2)        # [W, 2]    
       97 +    PA = wall_xy - A.unsqueeze(0)                  # [W, 2]   
       98 +    t  = ((PA * BA).sum(-1) / ((BA * BA).sum(-1) + 1e-8)).clam
          +p(0.1, 0.9)   # [W]                                           
       99 +    proj = A.unsqueeze(0) + t.unsqueeze(-1) * BA   # [W, 2]   
      100 +    dist = (wall_xy - proj).norm(dim=-1)            # [W]     
      101 +    return bool((dist < margin).any().item())                 
      102 +                                                              
      103 +                                                              
      104 +def _find_best_doorway(                                       
      105 +    curr_xy:  torch.Tensor,            # [2]                  
      106 +    goal_xy:  torch.Tensor,            # [2]                  
      107 +    door_xy:  Optional[torch.Tensor],  # [D, 2]  or None      
      108 +) -> Optional[torch.Tensor]:                                  
      109 +    """                                                       
      110 +    Among known doorway slots pick the one that minimises tota
          +l detour:                                                     
      111 +        cost = dist(agent → door) + dist(door → goal)         
      112 +    Returns [2] tensor or None if no doorways.                
      113 +    """                                                       
      114 +    if door_xy is None or door_xy.numel() == 0:               
      115 +        return None                                           
      116 +    device    = door_xy.device                                
      117 +    d_to_door = (door_xy - curr_xy.to(device)).norm(dim=-1)   
          + # [D]                                                        
      118 +    d_to_goal = (door_xy - goal_xy.to(device)).norm(dim=-1)   
          + # [D]                                                        
      119 +    best_idx  = int((d_to_door + d_to_goal).argmin().item())  
      120 +    return door_xy[best_idx]   # [2]                          
      121 +                                                              
      122 +                                                              
      123 +def _compute_wall_penalty(                                    
      124 +    final_xy:  torch.Tensor,            # [N, 2]              
      125 +    curr_xy:   torch.Tensor,            # [2]                 
      126 +    wall_xy:   Optional[torch.Tensor],  # [W, 2]  or None     
      127 +    margin:    float,                                         
      128 +    weight:    float,                                         
      129 +    device:    torch.device,                                  
      130 +) -> torch.Tensor:                                            
      131 +    """                                                       
      132 +    Vectorised Gaussian repulsion from wall slots.            
      133 +                                                              
      134 +    For each candidate sequence i, compute the closest approac
          +h of the path                                                 
      135 +    segment (curr_xy → final_xy[i]) to every wall slot j.  Sum
          + Gaussian                                                     
      136 +    penalties over all walls.                                 
      137 +                                                              
      138 +    Returns [N] penalty tensor (zero when no wall slots presen
          +t).                                                           
      139 +    """                                                       
      140 +    if wall_xy is None or wall_xy.numel() == 0:               
      141 +        return torch.zeros(final_xy.shape[0], device=device)  
      142 +                                                              
      143 +    N = final_xy.shape[0]                                     
      144 +    W = wall_xy.shape[0]                                      
      145 +                                                              
      146 +    A = curr_xy.to(device).unsqueeze(0).unsqueeze(0).expand(N,
          + W, 2)    # [N, W, 2]                                         
      147 +    B = final_xy.unsqueeze(1).expand(N, W, 2)                 
          +           # [N, W, 2]                                        
      148 +    P = wall_xy.to(device).unsqueeze(0).expand(N, W, 2)       
          +           # [N, W, 2]                                        
      149 +                                                              
      150 +    BA  = B - A                                               
          +             # [N, W, 2]                                      
      151 +    PA  = P - A                                               
          +             # [N, W, 2]                                      
      152 +    t   = ((PA * BA).sum(-1) / ((BA * BA).sum(-1) + 1e-8)).cla
          +mp(0, 1)  # [N, W]                                            
      153 +    proj = A + t.unsqueeze(-1) * BA                           
          +             # [N, W, 2]                                      
      154 +    dist = (P - proj).norm(dim=-1)                            
          +             # [N, W]                                         
      155 +                                                              
      156 +    penalty = weight * torch.exp(-dist ** 2 / (2.0 * margin **
          + 2))        # [N, W]                                          
      157 +    return penalty.sum(-1)                                    
          +              # [N]                                           
      158 +                                                              
      159 +                                                              
      160 +# ── Planner class ───────────────────────────────────────────
          +───────────────────                                           
      161 +                                                              
      162  class EFEPlannerV4:
      163      """
       51 -    Stateless planner — receives a belief and goal, returns th
          -e best action.                                                
      164 +    EFE planner for goal-directed navigation with structural-s
          +lot awareness.                                                
      165 +                                                              
      166 +    Stage 2 additions:                                        
      167 +      - Wall repulsion: penalises imagined paths that pass nea
          +r wall slots.                                                 
      168 +      - Doorway routing: when wall slots block the direct path
          + to goal, the                                                 
      169 +        effective goal is redirected to the best doorway slot.
          +  After the agent                                             
      170 +        passes through the doorway, the original goal is resto
          +red.                                                          
      171      """
      172  
      173      def __init__(
      174          self,
       56 -        model: WorldModelV6,                                  
      175 +        model:    WorldModelV6,                               
      176          goal_pos: Optional[torch.Tensor] = None,   # [2]  (x, 
           y) world frame
       58 -        cfg: Optional[EFEConfig] = None,                      
      177 +        cfg:      Optional[EFEConfig]    = None,              
      178      ):
      179          self.model    = model
       61 -        self.goal_pos = goal_pos   # set/update via set_goal()
      180          self.cfg      = cfg or model.cfg.efe
      181          self._candidates: Optional[torch.Tensor] = None
      182  
      183 +        # Effective goal (may temporarily point to a doorway) 
      184 +        self.goal_pos:     Optional[torch.Tensor] = None      
      185 +        # True final navigation target — never redirected     
      186 +        self._nav_goal:    Optional[torch.Tensor] = None      
      187 +        # Doorway-routing state                               
      188 +        self._via_door:    bool                   = False     
      189 +        self._door_target: Optional[torch.Tensor] = None      
      190 +                                                              
      191 +        if goal_pos is not None:                              
      192 +            self.set_goal(goal_pos)                           
      193 +                                                              
      194      # --------------------------------------------------------
           ---------- #
      195      # Goal management
      196      # --------------------------------------------------------
           ---------- #
      197  
      198      def set_goal(self, goal_pos: torch.Tensor) -> None:
       70 -        """Update goal position (x, y) in world frame."""     
       71 -        self.goal_pos = goal_pos.float()                      
      199 +        """Set (or update) the final navigation goal. Resets d
          +oor-routing state."""                                         
      200 +        goal_pos          = goal_pos.float()                  
      201 +        self.goal_pos     = goal_pos                          
      202 +        self._nav_goal    = goal_pos.clone()                  
      203 +        self._via_door    = False                             
      204 +        self._door_target = None                              
      205  
      206 +    @property                                                 
      207 +    def is_via_door(self) -> bool:                            
      208 +        """True while the effective goal is a doorway waypoint
          +, not the final goal."""                                      
      209 +        return self._via_door                                 
      210 +                                                              
      211      # --------------------------------------------------------
           ---------- #
      212 +    # Distance helpers                                        
      213 +    # --------------------------------------------------------
          +---------- #                                                  
      214 +                                                              
      215 +    def goal_distance(self, belief: BeliefState) -> float:    
      216 +        """Distance to the EFFECTIVE goal (doorway when routin
          +g, else final goal)."""                                       
      217 +        if self.goal_pos is None:                             
      218 +            return float("inf")                               
      219 +        return float((belief.pose_mu[:2] - self.goal_pos.to(be
          +lief.device)).norm().item())                                  
      220 +                                                              
      221 +    def nav_goal_distance(self, belief: BeliefState) -> float:
      222 +        """Distance to the FINAL navigation goal regardless of
          + door-routing state."""                                       
      223 +        ref = self._nav_goal if self._nav_goal is not None els
          +e self.goal_pos                                               
      224 +        if ref is None:                                       
      225 +            return float("inf")                               
      226 +        return float((belief.pose_mu[:2] - ref.to(belief.devic
          +e)).norm().item())                                            
      227 +                                                              
      228 +    def reached_goal(self, belief: BeliefState) -> bool:      
      229 +        """True when the FINAL navigation goal has been reache
          +d."""                                                         
      230 +        return self.nav_goal_distance(belief) <= self.cfg.goal
          +_radius                                                       
      231 +                                                              
      232 +    # --------------------------------------------------------
          +---------- #                                                  
      233      # Main planning interface
      234      # --------------------------------------------------------
           ---------- #
      235  
      236      @torch.no_grad()
      237      def select_action(
      238          self,
       80 -        belief:              BeliefState,                     
       81 -        recent_poses:        Optional[List[torch.Tensor]] = No
          -ne,                                                           
       82 -        blocked_headings:    Optional[Set[float]] = None,     
       83 -        explored_poses:      Optional[List[torch.Tensor]] = No
          -ne,  # all visited poses (exploration phase)                  
      239 +        belief:           BeliefState,                        
      240 +        recent_poses:     Optional[List[torch.Tensor]] = None,
      241 +        blocked_headings: Optional[Set[float]]         = None,
      242 +        explored_poses:   Optional[List[torch.Tensor]] = None,
      243      ) -> Dict[str, Any]:
      244          """
      245          Return the best first action for the current belief.
     ...
      249            best_sequence: [int, ...]
      250            best_efe     : float
      251            all_efe      : [N] tensor
      252 +          risk / ambiguity / info_gain: float (for logging)   
      253 +          wall_penalty : float (for logging)                  
      254 +          via_door     : bool  (True when routing through a do
          +orway)                                                        
      255          """
      256          device     = belief.device
      257          candidates = self._get_candidates(device)   # [N, K]
      258          N, K       = candidates.shape
      259  
      260 +        # ---- Extract structural slots from current belief --
          +--                                                            
      261 +        wall_xy, door_xy = _extract_structural_slots(belief, s
          +elf.model.cfg.slots)                                          
      262 +                                                              
      263 +        # ---- Doorway-routing mode management (with hysteresi
          +s) ----                                                       
      264 +        #                                                     
      265 +        # State transitions:                                  
      266 +        #   DIRECT → VIA_DOOR  : direct path to _nav_goal cros
          +ses a wall slot                                               
      267 +        #                         AND at least one doorway slo
          +t is known.                                                   
      268 +        #   VIA_DOOR → DIRECT  : agent is within door_pass_rad
          +ius of door target                                            
      269 +        #                         (door has been passed throug
          +h).                                                           
      270 +        if self._via_door and self._door_target is not None:  
      271 +            dist_to_door = (                                  
      272 +                belief.pose_mu[:2].to(device) - self._door_tar
          +get.to(device)                                                
      273 +            ).norm()                                          
      274 +            if dist_to_door <= self.cfg.door_pass_radius_m:   
      275 +                # Passed through the door — revert to original
          + goal                                                         
      276 +                self._via_door    = False                     
      277 +                self._door_target = None                      
      278 +                if self._nav_goal is not None:                
      279 +                    self.goal_pos = self._nav_goal            
      280 +        elif (                                                
      281 +            self._nav_goal is not None                        
      282 +            and not self._via_door                            
      283 +            and wall_xy is not None                           
      284 +        ):                                                    
      285 +            curr_xy = belief.pose_mu[:2].to(device)           
      286 +            if _is_path_blocked(curr_xy, self._nav_goal.to(dev
          +ice),                                                         
      287 +                                wall_xy, self.cfg.door_block_m
          +argin_m):                                                     
      288 +                best_door = _find_best_doorway(               
      289 +                    curr_xy, self._nav_goal.to(device), door_x
          +y                                                             
      290 +                )                                             
      291 +                if best_door is not None:                     
      292 +                    self._via_door    = True                  
      293 +                    self._door_target = best_door.to(device)  
      294 +                    self.goal_pos     = best_door.to(device)  
      295 +                                                              
      296          # ---- Batched imagination rollout ----
      297          roll = self.model.imagine_rollout(belief, candidates)
      298  
      101 -        pose_mu_final     = roll["pose_mu_final"]      # [N, 3
          -]                                                             
      102 -        ambiguity_roll    = roll["ambiguity_roll"]      # [N, 
          -K]                                                            
      103 -        info_gain_roll    = roll["info_gain_roll"]      # [N, 
          -K]                                                            
      299 +        pose_mu_final  = roll["pose_mu_final"]   # [N, 3]     
      300 +        ambiguity_roll = roll["ambiguity_roll"]   # [N, K]    
      301 +        info_gain_roll = roll["info_gain_roll"]   # [N, K]    
      302  
      303          # ---- Discount weights ----
      304          discount = torch.tensor(
     ...
      306              dtype=torch.float32, device=device,
      307          )  # [K]
      308  
      111 -        # ---- Risk: distance to goal at final step ----      
      309 +        # ---- Risk: distance to effective goal at final step 
          +----                                                          
      310          if self.goal_pos is not None:
      113 -            gp = self.goal_pos.to(device)                     
      311 +            gp      = self.goal_pos.to(device)                
      312              dist_sq = ((pose_mu_final[:, :2] - gp.unsqueeze(0)
           ) ** 2).sum(-1)  # [N]
      115 -            risk = dist_sq / (2.0 * self.cfg.goal_radius ** 2)
          -                  # [N]                                       
      313 +            risk    = dist_sq / (2.0 * self.cfg.goal_radius **
          + 2)               # [N]                                       
      314          else:
      315              risk = torch.zeros(N, device=device)
      316  
     ...
      321          info_gain = (discount.unsqueeze(0) * info_gain_roll).s
           um(-1)   # [N]
      322  
      323          # ---- Action cost (prefer forward over turning) ----
      126 -        fwd_count  = (candidates == 0).float().sum(-1)   # [N]
      324 +        fwd_count   = (candidates == 0).float().sum(-1)   # [N
          +]                                                             
      325          action_cost = self.cfg.w_action_cost * (K - fwd_count)
      326  
      327          # ---- Stay / loop penalty ----
     ...
      329          if recent_poses and len(recent_poses) >= 2:
      330              last_xy = recent_poses[-1].to(device)[:2]
      331              pred_xy = pose_mu_final[:, :2]
      134 -            near_last = ((pred_xy - last_xy.unsqueeze(0)).norm
          -(dim=-1) < 0.3).float()                                       
      332 +            near_last    = ((pred_xy - last_xy.unsqueeze(0)).n
          +orm(dim=-1) < 0.3).float()                                    
      333              stay_penalty = self.cfg.w_stay_penalty * near_last
      334  
      335          # ---- Wall-collision penalty ----
      138 -        # For each candidate sequence, simulate the heading at
          - the point the first                                          
      139 -        # FORWARD is attempted (after any leading turns).  If 
          -that heading is in                                            
      140 -        # blocked_headings (known wall), assign 1e6 to prevent
          - oscillation between                                          
      141 -        # two adjacent blocked headings.  The set clears when 
          -FORWARD succeeds.                                             
      336          blocked_penalty = torch.zeros(N, device=device)
      337          if blocked_headings:
      338              turn_step  = self.model.cfg.pose.turn_step_rad
     ...
      340              for i, seq in enumerate(candidates.tolist()):
      341                  theta = curr_theta
      342                  for act in seq:
      149 -                    if act == 0:   # FORWARD: check heading   
      343 +                    if act == 0:                              
      344                          h = _round_heading(theta, turn_step)
      345                          if h in blocked_headings:
      346                              blocked_penalty[i] = 1e6
      347                          break
      154 -                    elif act == 1:  # TURN_L                  
      348 +                    elif act == 1:                            
      349                          theta = _wrap(theta + turn_step)
      156 -                    elif act == 2:  # TURN_R                  
      350 +                    elif act == 2:                            
      351                          theta = _wrap(theta - turn_step)
      352  
      159 -        # If every sequence containing a FORWARD is blocked, d
          -isable the penalty                                            
      160 -        # so the planner can still pick a forward direction in
          -stead of oscillating                                          
      161 -        # between pure-turn sequences indefinitely.           
      353 +        # Disable penalty if ALL forward-containing sequences 
          +are blocked                                                   
      354          if blocked_headings:
      163 -            has_fwd = (candidates == 0).any(dim=-1)   # [N]   
      355 +            has_fwd = (candidates == 0).any(dim=-1)           
      356              if has_fwd.any() and (blocked_penalty[has_fwd] > 0
           ).all():
      357                  blocked_penalty = torch.zeros(N, device=device
           )
      358  
      359          # ---- Frontier penalty ----
      168 -        # During exploration (goal_pos=None), push the agent t
          -oward positions                                               
      169 -        # it has NOT visited yet.  For each candidate, compute
          - the minimum                                                  
      170 -        # Euclidean distance from its imagined final XY to any
          - explored pose.                                               
      171 -        # If that distance < frontier_radius, the position is 
          -"already known"                                               
      172 -        # and gets a penalty, forcing the planner to prefer no
          -vel areas.                                                    
      360          frontier_penalty = torch.zeros(N, device=device)
      361          if explored_poses and len(explored_poses) > 0:
      175 -            exp_xy = torch.stack(                             
      176 -                [p.to(device)[:2] for p in explored_poses], di
          -m=0                                                           
      177 -            )  # [M, 2]                                       
      178 -            pred_xy = pose_mu_final[:, :2]             # [N, 2
          -]                                                             
      179 -            # cdist: [N, M] pairwise distances                
      180 -            dists    = torch.cdist(pred_xy, exp_xy)    # [N, M
          -]                                                             
      181 -            min_dist = dists.min(dim=-1).values         # [N] 
      362 +            exp_xy   = torch.stack([p.to(device)[:2] for p in 
          +explored_poses], dim=0)                                       
      363 +            pred_xy  = pose_mu_final[:, :2]                   
      364 +            dists    = torch.cdist(pred_xy, exp_xy)           
      365 +            min_dist = dists.min(dim=-1).values               
      366              near     = (min_dist < self.cfg.frontier_radius).f
           loat()
      367              frontier_penalty = self.cfg.w_frontier_penalty * n
           ear
      368  
      369          # ---- Inverse-pair penalty ----
      186 -        # Sequences that open with L→R or R→L waste two steps 
          -on a do-undo turn.                                            
      187 -        # The planner exploits them near the goal (the K-2 FOR
          -WARDs that follow                                             
      188 -        # end up close to the target), but executing one step 
          -at a time means the                                           
      189 -        # agent never commits: it alternates between the two t
          -urns forever.                                                 
      370          inverse_pair_penalty = torch.zeros(N, device=device)
      371          if K >= 2:
      192 -            first  = candidates[:, 0]   # [N]                 
      193 -            second = candidates[:, 1]   # [N]                 
      372 +            first  = candidates[:, 0]                         
      373 +            second = candidates[:, 1]                         
      374              bad = ((first == 1) & (second == 2)) | ((first == 
           2) & (second == 1))
      375              inverse_pair_penalty[bad] = self.cfg.w_inverse_pai
           r_penalty
      376  
      377 +        # ---- Wall proximity penalty (Stage 2) ----          
      378 +        # Penalise any candidate whose imagined path segment (
          +curr → final_xy)                                              
      379 +        # passes close to a known wall slot.  Works in both ph
          +ases.                                                         
      380 +        wall_penalty = _compute_wall_penalty(                 
      381 +            pose_mu_final[:, :2],                             
      382 +            belief.pose_mu[:2].to(device),                    
      383 +            wall_xy,                                          
      384 +            self.cfg.wall_margin_m,                           
      385 +            self.cfg.w_wall_penalty,                          
      386 +            device,                                           
      387 +        )                                                     
      388 +                                                              
      389          # ---- Total EFE ----
      390          efe = (
      391              self.cfg.w_risk       * risk
     ...
      396              + blocked_penalty
      397              + inverse_pair_penalty
      398              + frontier_penalty
      399 +            + wall_penalty                                    
      400          )
      401  
      402          best_idx = int(efe.argmin().item())
     ...
      410              "risk":          float(risk[best_idx].item()),
      411              "ambiguity":     float(ambiguity[best_idx].item())
           ,
      412              "info_gain":     float(info_gain[best_idx].item())
           ,
      413 +            "wall_penalty":  float(wall_penalty[best_idx].item
          +()),                                                          
      414 +            "via_door":      self._via_door,                  
      415          }
      416  
      417      # --------------------------------------------------------
           ---------- #
     ...
      427          seqs = list(itertools.product(range(A), repeat=K))
      428          self._candidates = torch.tensor(seqs, dtype=torch.long
           )
      429          return self._candidates.to(device)
      235 -                                                              
      236 -    # --------------------------------------------------------
          ----------- #                                                  
      237 -    # Distance to goal (for external logging)                 
      238 -    # --------------------------------------------------------
          ----------- #                                                  
      239 -                                                              
      240 -    def goal_distance(self, belief: BeliefState) -> float:    
      241 -        """Euclidean distance from current pose estimate to go
          -al."""                                                        
      242 -        if self.goal_pos is None:                             
      243 -            return float("inf")                               
      244 -        return float(                                         
      245 -            (belief.pose_mu[:2] - self.goal_pos.to(belief.devi
          -ce)).norm().item()                                            
      246 -        )                                                     
      247 -                                                              
      248 -    def reached_goal(self, belief: BeliefState) -> bool:      
      249 -        return self.goal_distance(belief) <= self.cfg.goal_rad
          -ius                                                           

● Now update the demo to use nav_goal_distance for final-goal tracking and log
  door-routing state:

  Read 1 file (ctrl+o to expand)

● Now I'll make the targeted demo changes: use nav_goal_distance in summary, add
  door-routing state logging:

● Update(aif_explore_nav_demo.py)
  ⎿  Added 1 line
      384      consecutive_non_fwd = 0
      385      wall_avoid_queue: List[int] = []
      386      nav_phase_attempted = False   # ensure phase switch is tri
           ed exactly once
      387 +    prev_via_door = False          # track door-routing state 
          +transitions for logging                                       
      388      reached = False
      389  
      390      log(f"\n{'Step':>4} | {'Phase':>8} | {'Action':>7} | "

● Update(aif_explore_nav_demo.py)
  ⎿  Added 12 lines, removed 1 line
      522                  blocked_headings = blocked_headings,
      523                  explored_poses   = explored_poses if phase == 
           "EXPLORE" else None,
      524              )
      525 -        action = plan["best_action"]                          
      525 +        action       = plan["best_action"]                    
      526 +        via_door_now = plan.get("via_door", False)            
      527 +        wall_pen_val = plan.get("wall_penalty", 0.0)          
      528  
      529 +        # Log door-routing mode transitions                   
      530 +        if via_door_now and not prev_via_door and planner._doo
          +r_target is not None:                                         
      531 +            dt = planner._door_target                         
      532 +            log(f"       [VIA-DOOR activated → doorway at "   
      533 +                f"({float(dt[0]):.2f}, {float(dt[1]):.2f})]") 
      534 +        elif not via_door_now and prev_via_door:              
      535 +            log(f"       [VIA-DOOR: passed through — reverting
          + to final nav goal]")                                         
      536 +        prev_via_door = via_door_now                          
      537 +                                                              
      538          # Priority 1: Wall-escape queue (both phases, highest 
           priority)
      539          if wall_avoid_queue:
      540              action = wall_avoid_queue.pop(0)

● Update(aif_explore_nav_demo.py)
  ⎿  Added 4 lines, removed 1 line
      620          x, y, th = belief.pose_mu.tolist()
      621          ms = (time.perf_counter() - t0) * 1000
      622          step_times_ms.append(ms)
      623 +        door_tag = " [VIA-DOOR]" if via_door_now else ""      
      624          log(f"{step+1:>4} | {phase:>8} | {ACTION_NAMES[action]
           :>7} | "
      625              f"({x:6.2f}, {y:6.2f}, {math.degrees(th):5.1f}°) |
            "
      625 -            f"{goal_dist:8.3f} | {n_slots:>5} | {ms:5.0f}")   
      626 +            f"{goal_dist:8.3f} | {n_slots:>5} | {ms:5.0f}{door
          +_tag}")                                                       
      627 +        if wall_pen_val > 1.0:                                
      628 +            log(f"       [wall-penalty={wall_pen_val:.2f}]")  
      629  
      630          # ── render frame ────────────────────────────────────
           ──────────────────
      631          rgb_np = rgb.permute(1, 2, 0).cpu().numpy()

● Update(aif_explore_nav_demo.py)
  ⎿  Added 1 line, removed 1 line
      719              break
      720  
      721      # ── summary ─────────────────────────────────────────────
           ──────────────────
      722 -    final_dist = planner.goal_distance(belief) if phase == "NA
          -VIGATE" else float("nan")                                     
      722 +    final_dist = planner.nav_goal_distance(belief) if phase ==
          + "NAVIGATE" else float("nan")                                 
      723      log()
      724      log("=" * 60)
      725      log(f"  Result         : {'SUCCESS' if reached else 'TIMEO
           UT'}")

● Now let me do a quick sanity check — verify the new planner imports and the
  config fields:

  Ran 3 shell commands (ctrl+o to expand)

● All helpers pass. Let me verify the doorway-routing mode switch with a
  synthetic scenario:

  Ran 1 shell command (ctrl+o to expand)

● The routing switch works perfectly. Let me also verify the "pass-through door"
  reversion:

  Ran 1 shell command (ctrl+o to expand)

● Both transitions work. Now run the full demo to see Stage 2 in action:
  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● Let me check the end of the run and look for any door-routing events:

  Read 1 file (ctrl+o to expand)

● The run succeeded. Let me look at why the exploration barely moved (the agent
  seems to stay near x≈2.0 throughout):

  Searched for 1 pattern (ctrl+o to expand)

● Stage 2 implemented and verified. Let me update the memory file to reflect the
  current state:

  Recalled 1 memory, wrote 1 memory (ctrl+o to expand)

● Stage 2 is complete. Here's what was implemented and what the live run showed:

  ---
  Stage 2 — what changed

  planner/efe_planner_v4.py — two new EFE terms:

  1. Wall proximity penalty (both phases, always active)
  Vectorised line-segment distance from each imagined path (curr→final_xy) to
  every wall slot, summed as Gaussians. The live run showed 3–13 penalty units
  throughout navigation — meaningfully steering the agent away from detected
  walls while not overriding final-approach.
  2. Doorway routing (navigation, when direct path is blocked)
  Hysteresis state machine: when _is_path_blocked(curr, goal, wall_slots, 
  margin=0.7m) fires AND a doorway slot exists, goal_pos is silently redirected
  to the best doorway (minimises detour cost). The original nav goal is
  preserved in _nav_goal. After the agent gets within 0.8m of the doorway, it
  reverts automatically.
  reached_goal() and nav_goal_distance() always check _nav_goal so success
  detection is correct regardless of routing state.

  Live run result — SUCCESS in 100 steps (80 explore + 20 navigate), final dist
  = 0.337m at 96ms/step. Door-routing was NOT triggered because the Habitat
  pathfinder waypoints already route around walls — correct behavior. It was
  validated separately with a synthetic belief (wall at (2,0), doorway at
  (2,1.2), goal at (4,0)) where it activated immediately and reverted after
  passing through.

  Observation about exploration stuck issue: The exploration phase kept the
  agent near its starting position because waypoint-0 direction (≈180°) had a
  wall collision at step 8 — the heading controller keeps trying but hits the
  same wall. This is a pre-existing issue that Stage 3 (wall-following
  exploration) would fix.

✻ Worked for 13m 30s

❯ start implementing stage 3                                                    

● Let me read the current planner __init__.py and the demo file's explore
  section before writing Stage 3:

  Read 1 file, listed 1 directory (ctrl+o to expand)

● Write(~/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/planner/wall_follo
       w_explorer.py)
  ⎿  Wrote 260 lines to ../planner/wall_follow_explorer.py
       1 """
       3 
       4 Replaces random-waypoint exploration (Priority 2 during EXPLORE 
         phase).
       6   1. SEEK  — move forward in that heading until wall contact or 
         max_seek steps.
         l circuit or time-out.
       8 After all 4 headings: DONE → caller falls through to EFE info-ga
         in.
      10 Design:
     … +298 lines (ctrl+o to expand)

● Update(~/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/planner/__init__.
        py)
  ⎿  Added 2 lines, removed 1 line
      1  from .efe_planner_v4 import EFEPlannerV4
      2 -__all__ = ["EFEPlannerV4"]                                      
      2 +from .wall_follow_explorer import WallFollowExplorer            
      3 +__all__ = ["EFEPlannerV4", "WallFollowExplorer"]                

● Now update the demo to wire in Stage 3 — replace random-waypoint Priority 2
  with the wall-follower:

  Read 1 file (ctrl+o to expand)

● Update(aif_explore_nav_demo.py)
  ⎿  Added 11 lines, removed 11 lines
       3  
       4  PHASE 1 — EXPLORATION  (no goal given)
       5    EFE = Ambiguity − InfoGain + FrontierPenalty
       6 -  The agent explores the apartment using random waypoint explor
         -ation:                                                         
       7 -    • N_EXPLORE_WPS random navigable positions are sampled via 
         -the Habitat                                                    
       8 -      pathfinder at episode start (wall-aware, adapts to apartm
         -ent geometry).                                                 
       9 -    • A heading controller navigates to each waypoint sequentia
         -lly.                                                           
      10 -    • InfoGain + FrontierPenalty continue to shape EFE as fallb
         -ack.                                                           
      11 -  As the agent moves, its 64-slot belief state fills with detec
         -ted object                                                     
      12 -  positions (chair, sofa, TV, …).  This IS the object-centric m
         -ap.                                                            
       6 +  The agent explores using a wall-following state machine (Stag
         +e 3):                                                          
       7 +    • Cycles through 4 cardinal headings (E, N, W, S).         
       8 +    • For each heading: SEEK (move forward until wall) → FOLLOW
         + (right-hand                                                   
       9 +      wall-following rule, discovering gaps/doorways naturally)
         +.                                                              
      10 +    • After all 4 headings: EFE info-gain takes over as fallbac
         +k.                                                             
      11 +  As the agent moves, 64-slot belief fills with walls, doorways
         +, and objects.                                                 
      12  
      13  PHASE 2 — NAVIGATION   (goal provided after exploration)
      15 -  EFE = Risk + Ambiguity − InfoGain                            
      14 +  EFE = Risk + Ambiguity − InfoGain + WallPenalty + DoorwayRout
         +ing (Stage 2)                                                  
      15    The belief state (object map) is carried over from Phase 1.
      16    A geodesically reachable goal is sampled; the Habitat pathfin
          der provides
      18 -  waypoints.  The EFE planner uses accumulated slot landmarks f
         -or localisation                                                
      19 -  while Risk pulls it toward each successive waypoint.         
      17 +  waypoints.  The EFE planner uses accumulated slot landmarks +
         + wall/doorway                                                  
      18 +  slots for topology-aware navigation.                         
      19  
      20  VIDEO  results/aif_explore_nav_YYYYMMDD_HHMMSS.mp4
      21    Left : first-person RGB with phase / step / goal dist annotat
          ion.
     ...
      23           • Navmesh background
      24           • Orange dotted line  : planned waypoint route
      25           • Blue dots + line    : path actually taken
      27 -         • Coloured scatter    : slot belief map (each object s
         -lot as a dot)                                                  
      26 +         • Coloured scatter    : slot belief map               
      27 +             red squares = walls, green triangles = doorways, c
         +ircles = objects                                               
      28           • Green dot + arrow   : current robot pose
      29           • Gold ★              : start position
      30           • Red ★ + dashed ring : goal (navigation phase only)

● Update(aif_explore_nav_demo.py)
  ⎿  Added 1 line, removed 4 lines
      53  sys.path.insert(0, _V6_ROOT)
      54  
      55  from model_v6 import WorldModelV6, ModelV6Config, BeliefState, 
          WALL_ID, DOORWAY_ID
      56 -from planner  import EFEPlannerV4                              
      56 +from planner  import EFEPlannerV4, WallFollowExplorer          
      57  from habitat_adapter.env_wrapper import HabitatEnv
      58  
      59  ACTION_NAMES   = {0: "FORWARD", 1: "TURN_L", 2: "TURN_R", 3: "S
          TOP"}
      60  MAX_STEPS      = 200          # total budget (explore + navigat
          e)
      61  N_EXPLORE      = 80           # steps in exploration phase
      62 -N_EXPLORE_WPS  = 5            # random navigable waypoints to v
         -isit during exploration                                        
      63 -EXPLORE_WP_MIN = 1.0          # min geodesic dist (m) for each 
         -explore waypoint                                               
      64 -EXPLORE_WP_MAX = 3.0          # max geodesic dist (m) for each 
         -explore waypoint                                               
      62  GOAL_RADIUS          = 0.5    # metres — success zone
      63  FINAL_APPROACH_DIST  = 1.5    # metres — switch to direct headi
          ng+forward controller
      64  WP_ADVANCE     = 0.8          # metres — advance to next waypoi
          nt when this close

● Now update _render_frame — remove explore-waypoint parameters, add
  wall-follower phase display:

● Update(aif_explore_nav_demo.py)
  ⎿  Added 1 line, removed 2 lines
      120      goal_v6:      Optional[torch.Tensor] = None,
      121      waypoints_v6: Optional[List[torch.Tensor]] = None,
      122      wp_idx:       int = 0,
      123 -    explore_wps_v6: Optional[List[torch.Tensor]] = None,      
      124 -    explore_wp_idx: int = 0,                                  
      123 +    wf_phase:     str = "",                     # wall-followe
          +r phase tag for display                                       
      124  ) -> np.ndarray:
      125  
      126      fig, (ax_rgb, ax_map) = plt.subplots(

● Now update the map title and remove the explore-waypoints rendering block:

● Update(aif_explore_nav_demo.py)
  ⎿  Removed 12 lines
      208                             edgecolors="darkgreen", linewidths=
           0.6,
      209                             zorder=6, label=f"{int(doorway_mask
           .sum())} doors")
      210  
      211 -    # Exploration waypoints (exploration phase only)          
      212 -    if phase == "EXPLORE" and explore_wps_v6:                 
      213 -        for i, wp in enumerate(explore_wps_v6):               
      214 -            wx_h, wz_h = _v6_to_hab(float(wp[0]), float(wp[1])
          -)                                                             
      215 -            ax_map.scatter([wx_h], [wz_h], s=18, c="cyan", alp
          -ha=0.55, zorder=4)                                            
      216 -        if explore_wp_idx < len(explore_wps_v6):              
      217 -            cwp = explore_wps_v6[explore_wp_idx]              
      218 -            cwx_h, cwz_h = _v6_to_hab(float(cwp[0]), float(cwp
          -[1]))                                                         
      219 -            ax_map.scatter([cwx_h], [cwz_h], s=90, c="cyan", m
          -arker="D",                                                    
      220 -                           edgecolors="black", lw=0.5, zorder=
          -8,                                                            
      221 -                           label=f"exp-wp[{explore_wp_idx}]") 
      222 -                                                              
      211      # Start marker
      212      sx_h, sz_h = _v6_to_hab(*start_v6)
      213      ax_map.scatter([sx_h], [sz_h], s=120, c="gold", marker="*"
           ,

● Update(aif_explore_nav_demo.py)
  ⎿  Added 5 lines, removed 2 lines
      243                      arrowprops=dict(arrowstyle="->", color="da
           rkgreen", lw=1.8),
      244                      zorder=10)
      245  
      246 -    phase_label = "Exploration  (EFE = Ambiguity - InfoGain)" 
          -\                                                             
      247 -        if phase == "EXPLORE" else "Navigation  (EFE = Risk + 
          -Ambiguity - InfoGain)"                                        
      246 +    if phase == "EXPLORE":                                    
      247 +        phase_label = f"Exploration — {wf_phase}  (EFE=Ambigui
          +ty−InfoGain)" if wf_phase \                                   
      248 +                       else "Exploration  (EFE=Ambiguity−InfoG
          +ain, WF-DONE)"                                                
      249 +    else:                                                     
      250 +        phase_label = "Navigation  (EFE=Risk+Ambiguity−InfoGai
          +n+WallPenalty)"                                               
      251      ax_map.set_title(f"Object-centric belief map  |  {phase_la
           bel}",
      252                       fontsize=7.5, color=phase_color)
      253      ax_map.set_xlabel("Habitat X", fontsize=6)

● Now remove the _sample_explore_waypoints function and update the main loop
  setup section:

● Update(aif_explore_nav_demo.py)
  ⎿  Removed 28 lines
      266      return frame
      267  
      268  
      269 -# ── exploration waypoint sampler ────────────────────────────
          -──────────────────                                            
      270 -                                                              
      271 -def _sample_explore_waypoints(                                
      272 -    env,                                                      
      273 -    n_wps:    int   = N_EXPLORE_WPS,                          
      274 -    min_dist: float = EXPLORE_WP_MIN,                         
      275 -    max_dist: float = EXPLORE_WP_MAX,                         
      276 -    log = print,                                              
      277 -) -> List[torch.Tensor]:                                      
      278 -    """                                                       
      279 -    Sample random navigable positions to guide the exploration
          - phase.                                                       
      280 -                                                              
      281 -    Uses Habitat's pathfinder to snap candidates onto the navm
          -esh, so sampled                                               
      282 -    goals respect actual apartment geometry (rooms, corridors,
          - walls).  Each                                                
      283 -    returned tensor is a V6 [x_fwd, y_left] position at geodes
          -ic distance                                                   
      284 -    [min_dist, max_dist] from the current agent position.     
      285 -                                                              
      286 -    Returns a list of up to n_wps goal tensors.  The caller fo
          -llows them                                                    
      287 -    sequentially with a heading controller, advancing when wit
          -hin WP_ADVANCE.                                               
      288 -    """                                                       
      289 -    goals: List[torch.Tensor] = []                            
      290 -    for i in range(n_wps):                                    
      291 -        g = env.sample_goal(min_dist=min_dist, max_dist=max_di
          -st)                                                           
      292 -        goals.append(g)                                       
      293 -        log(f"  [explore-wp {i}: ({g[0]:.2f}, {g[1]:.2f})]")  
      294 -    return goals                                              
      295 -                                                              
      296 -                                                              
      269  # ── main ────────────────────────────────────────────────────
           ──────────────────
      270  
      271  def main():

● Now update the episode setup inside main() — remove waypoint sampling, add
  wall-follower:

● Update(aif_explore_nav_demo.py)
  ⎿  Added 7 lines, removed 8 lines
      313      # Exploration: no goal → Risk = 0, EFE driven by Ambiguity
            - InfoGain
      314      planner = EFEPlannerV4(model=model, goal_pos=None, cfg=cfg
           .efe)
      315  
      316 -    # ── sample exploration waypoints (from start position, be
          -fore first step) ─                                            
      317 -    log("\nSampling exploration waypoints...")                
      318 -    explore_wps: List[torch.Tensor] = _sample_explore_waypoint
          -s(                                                            
      319 -        env, N_EXPLORE_WPS, EXPLORE_WP_MIN, EXPLORE_WP_MAX, lo
          -g=log                                                         
      316 +    # ── wall-following explorer (Stage 3) ───────────────────
          +──────────────────                                            
      317 +    wall_follower = WallFollowExplorer(                       
      318 +        turn_step   = cfg.pose.turn_step_rad,                 
      319 +        max_seek    = 20,   # forward steps per heading before
          + giving up                                                    
      320 +        max_follow  = 80,   # max steps per wall-follow segmen
          +t                                                             
      321 +        return_dist = 0.8,  # metres — "returned to start" thr
          +eshold                                                        
      322      )
      321 -    explore_wp_idx    = 0                                     
      322 -    explore_current_wp: Optional[torch.Tensor] = (            
      323 -        explore_wps[0].to(device) if explore_wps else None    
      324 -    )                                                         
      323 +    log("\nWall-follow explorer ready  (SEEK E → N → W → S)") 
      324  
      325      # ── shared loop state ───────────────────────────────────
           ──────────────────
      326      prev_action   = 3
● Now update the shared loop state section — remove explore-waypoint variables:

● Update(aif_explore_nav_demo.py)
  ⎿  Added 1 line
      329      explored_poses: List[torch.Tensor] = []        # all visit
           ed positions (frontier)
      330      path_xy: List[Tuple[float, float]] = [start_xy]
      331      step_times_ms: List[float] = []
      332 +    collided      = False   # result of the previous step's ac
          +tion                                                          
      333  
      334      # Navigation-phase state (set at phase switch)
      335      phase         = "EXPLORE"

● Now update the main loop body — remove the explore-waypoint advance block and
  replace Priority 2:

● Update(aif_explore_nav_demo.py)
  ⎿  Added 1 line, removed 13 lines
      375              if len(explored_poses) > 300:
      376                  explored_poses = explored_poses[-300:]
      377  
      378 -        # ── explore waypoint advance ────────────────────────
          -──────────────────                                            
      379 -        if phase == "EXPLORE" and explore_current_wp is not No
          -ne:                                                           
      380 -            ew_dist = float((pose_v6[:2].to(device) - explore_
          -current_wp).norm())                                           
      381 -            if ew_dist < WP_ADVANCE:                          
      382 -                explore_wp_idx += 1                           
      383 -                if explore_wp_idx < len(explore_wps):         
      384 -                    explore_current_wp = explore_wps[explore_w
          -p_idx].to(device)                                             
      385 -                    blocked_headings.clear()                  
      386 -                    log(f"  >> explore-wp[{explore_wp_idx}/{le
          -n(explore_wps)-1}]: "                                         
      387 -                        f"({explore_current_wp[0]:.2f}, {explo
          -re_current_wp[1]:.2f})")                                      
      388 -                else:                                         
      389 -                    explore_current_wp = None                 
      390 -                    log(f"  >> All explore waypoints visited —
          - EFE info-gain takes over.")                                  
      378 +        # (no waypoint-advance logic — wall-follower manages i
          +ts own state)                                                 
      379  
      380          # ── PHASE SWITCH: exploration → navigation ──────────
           ──────────────────
      381          # Attempted exactly once (nav_phase_attempted flag), a
           lways succeeds:

● Now replace Priority 2 (old heading controller) with the wall-follower:

● Update(aif_explore_nav_demo.py)
  ⎿  Added 6 lines, removed 20 lines
      487              action = wall_avoid_queue.pop(0)
      488              log(f"       [forced: {ACTION_NAMES[action]}, {len
           (wall_avoid_queue)} left]")
      489  
      490 -        # Priority 2: Heading controller toward current explor
          -ation waypoint (EXPLORE only)                                 
      491 -        elif phase == "EXPLORE" and explore_current_wp is not 
          -None:                                                         
      492 -            dx_e    = float(explore_current_wp[0] - pose_v6[0]
          -)                                                             
      493 -            dy_e    = float(explore_current_wp[1] - pose_v6[1]
          -)                                                             
      494 -            exp_dir = math.atan2(dy_e, dx_e)                  
      495 -            curr_h  = float(pose_v6[2])                       
      496 -            herr    = math.atan2(math.sin(exp_dir - curr_h),  
      497 -                                  math.cos(exp_dir - curr_h)) 
      498 -            _ts = cfg.pose.turn_step_rad                      
      499 -            _rh = math.atan2(math.sin(round(curr_h / _ts) * _t
          -s),                                                           
      500 -                              math.cos(round(curr_h / _ts) * _
          -ts))                                                          
      501 -            if abs(herr) <= math.radians(30) and _rh not in bl
          -ocked_headings:                                               
      502 -                action = 0   # FORWARD                        
      503 -            elif herr > 0:                                    
      504 -                action = 1   # TURN_L                         
      505 -            else:                                             
      506 -                action = 2   # TURN_R                         
      507 -            ew_dist_log = float((pose_v6[:2].to(device) - expl
          -ore_current_wp).norm())                                       
      508 -            log(f"       [→wp{explore_wp_idx}: d={ew_dist_log:
          -.1f}m"                                                        
      509 -                f" h={math.degrees(herr):.0f}° → {ACTION_NAMES
          -[action]}]")                                                  
      490 +        # Priority 2: Wall-following explorer (EXPLORE phase, 
          +Stage 3)                                                      
      491 +        elif phase == "EXPLORE" and not wall_follower.done:   
      492 +            wf_act = wall_follower.get_action(pose_v6, collide
          +d)                                                            
      493 +            if wf_act is not None:                            
      494 +                action = wf_act                               
      495 +                log(f"       [{wall_follower.phase_name} → {AC
          +TION_NAMES[action]}]")                                        
      496  
      497          # Priority 3 — Final approach (navigation only): bypas
           s EFE within
      498          # FINAL_APPROACH_DIST. The K=5 imagination horizon ove
           rshoots the goal

● Now move collided into a persistent variable and fix the escape logic. Read
  the current execute + bookkeeping section:

  Read 1 file (ctrl+o to expand)

● Now fix the render_frame call and escape logic, and the GOAL REACHED frame:

  Read 1 file (ctrl+o to expand)

● Update(aif_explore_nav_demo.py)
  ⎿  Added 1 line, removed 1 line
      566              rgb_np, belief, cfg, step+1, phase, ACTION_NAMES[a
           ction], goal_dist,
      567              path_xy, start_xy, topdown, bounds,
      568              goal_v6=goal_v6, waypoints_v6=waypoints_v6, wp_idx
           =wp_idx,
      569 -            explore_wps_v6=explore_wps, explore_wp_idx=explore
          -_wp_idx,                                                      
      569 +            wf_phase=wall_follower.phase_name if phase == "EXP
          +LORE" else "",                                                
      570          )
      571          writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
      572  

  Read 1 file (ctrl+o to expand)

● Now fix the GOAL REACHED frame render and the stuck-escape logic:

  Read 1 file (ctrl+o to expand)

● Update(aif_explore_nav_demo.py)
  ⎿  Removed 1 line
      450                      rgb_np, belief, cfg, step+1, "NAVIGATE", "
           REACHED",
      451                      0.0, path_xy, start_xy, topdown, bounds,
      452                      goal_v6=goal_v6, waypoints_v6=waypoints_v6
           , wp_idx=wp_idx,
      453 -                    explore_wps_v6=explore_wps, explore_wp_idx
          -=explore_wp_idx,                                              
      453                  )
      454                  bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      455                  for _ in range(VID_FPS * 2):

● Now fix the escape logic and progress tracking for the wall-follower:

● Update(aif_explore_nav_demo.py)
  ⎿  Added 6 lines, removed 4 lines
      589                  blocked_headings.discard(_rh)
      590  
      591          # Track how many steps since meaningful forward progre
           ss.
      592 -        # Only wall-escape turns are exempt; explore heading-c
          -ontroller turns count.                                        
      592 +        # Wall-follower turns are intentional — suppress escap
          +e counter while active.                                       
      593          if action == 0 and not collided:
      594              steps_no_progress = 0
      595          elif action != 0 and wall_avoid_queue:
      596              pass   # wall-escape turn — don't count as stuck
      597 +        elif phase == "EXPLORE" and not wall_follower.done:   
      598 +            steps_no_progress = 0   # wall-follower owns its o
          +wn recovery                                                   
      599          else:
      600              steps_no_progress += 1
      601  
     ...
      608              prev_goal_dist = goal_dist
      609  
      610          # ── escape when stuck (BOTH phases) ─────────────────
           ─────────────────
      611 +        # During EXPLORE: only fire when wall-follower is DONE
          + (EFE fallback mode).                                         
      612          if steps_no_progress >= 8 and not wall_avoid_queue:
      613              if phase == "NAVIGATE" and blocked_headings and pl
           anner.goal_pos is not None:
      614                  # Navigation: perpendicular detour around wall
            (use GT pose)
     ...
      626                  wall_avoid_queue = [_td] * _nt + [0, 0, 0]
      627                  log(f"       [wall-escape: {_nt} turns to "
      628                      f"{math.degrees(_pdir):.0f}° + 3 FWD]")
      626 -            elif phase == "EXPLORE":                          
      627 -                # Exploration: sweep 90° left then try moving 
          -forward                                                       
      628 -                # This guarantees the camera scans a new direc
          -tion                                                          
      629 +            elif phase == "EXPLORE" and wall_follower.done:   
      630 +                # EFE fallback only — sweep if EFE gets stuck 
      631                  wall_avoid_queue = [1, 1, 1, 0, 0, 0]   # 3×TU
           RN_L + 3×FORWARD
      632                  log(f"       [explore-sweep: 3 TURN_L + 3 FORW
           ARD queued]")
      633              if blocked_headings:

● Quick sanity check — import test first, then run:

  Ran 1 shell command (ctrl+o to expand)

● Logic checks out. Now run the full demo with Stage 3:

  Searched for 1 pattern, read 1 file, ran 1 shell command (ctrl+o to expand)

● Stage 3 succeeded. Let me check the exploration trajectory to see how much
  area was covered:

  Searched for 4 patterns, ran 3 shell commands (ctrl+o to expand)

● Door-routing fired in a real navigation scenario. The agent was navigating
  west toward a goal at (-4.87, 0.48) from x≈-0.3, and there were wall slots
  between it and the goal. The planner detected a doorway at (-1.78, 0.16) and
  routed through it — then passed through, reverted, found the next doorway at
  (-2.01, 0.14), routed again. Multiple door-routing transitions in one episode.

  Let me look at the coverage improvement summary:

  Searched for 1 pattern (ctrl+o to expand)

● Stage 3 is working and producing strong results. Let me update the memory:

  Wrote 1 memory (ctrl+o to expand)

● Stage 3 is implemented and running. Here's what the live run showed:

  ---
  Stage 3 results — what changed and what it produced:

  WallFollowExplorer state machine (planner/wall_follow_explorer.py):
  - 4 headings: E → N → W → S
  - SEEK: move forward until wall, FOLLOW: right-hand rule with probe every 4
  steps

  Exploration coverage (dramatic improvement):

  ┌──────────────────┬───────────┬────────────┬──────────────────────────┐
  │      Metric      │ Lawnmower │ Random WPs │  Stage 3 (wall-follow)   │
  ├──────────────────┼───────────┼────────────┼──────────────────────────┤
  │ X span           │ 0.24m     │ 2.49m      │ 8.46m (3.4× improvement) │
  ├──────────────────┼───────────┼────────────┼──────────────────────────┤
  │ Slots (80 steps) │ 8-10      │ 21         │ 38                       │
  └──────────────────┴───────────┴────────────┴──────────────────────────┘

  Door-routing fired 11 times during navigation — the wall slots discovered
  during wall-following actually guided the agent through doorways between rooms
  on the way to the goal. Stage 2 + Stage 3 are now working together.

  One tuning issue: max_follow=80 equals N_EXPLORE=80, so only heading E was
  explored. The agent traced the east wall for all 80 steps. Reducing max_follow
  to ~30 would let it hit 2-3 headings instead. You can change this in the
  WallFollowExplorer(...) instantiation in main() — no code refactoring needed.
✻ Worked for 14m 5s

❯ can you the necessary changes and after that tell me which code I should run  
  to visualise the navigation                                                   

● Update(aif_explore_nav_demo.py)
  ⎿  Added 2 lines, removed 2 lines
      316      # ── wall-following explorer (Stage 3) ───────────────────
           ──────────────────
      317      wall_follower = WallFollowExplorer(
      318          turn_step   = cfg.pose.turn_step_rad,
      319 -        max_seek    = 20,   # forward steps per heading before
          - giving up                                                    
      320 -        max_follow  = 80,   # max steps per wall-follow segmen
          -t                                                             
      319 +        max_seek    = 15,   # forward steps per heading before
          + giving up                                                    
      320 +        max_follow  = 30,   # max steps per wall-follow segmen
          +t (allows ~2-3 headings in 80 steps)                          
      321          return_dist = 0.8,  # metres — "returned to start" thr
           eshold
      322      )
      323      log("\nWall-follow explorer ready  (SEEK E → N → W → S)")

● Done. To run and visualise:

  conda activate habitat
  cd /home/nitish/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/habitat_ad
  apter
  python aif_explore_nav_demo.py

  The script writes two files to Symbolic_AIF_v6/results/:
  - aif_explore_nav_YYYYMMDD_HHMMSS.mp4 — the video (open this to visualise)
  - aif_explore_nav_YYYYMMDD_HHMMSS.txt — the step-by-step log

  The MP4 has two panels side-by-side:
  - Left: first-person RGB camera with phase/step/goal-dist overlay
  - Right: top-down belief map — red squares = walls, green triangles =
  doorways, coloured circles = objects, green arrow = robot pose, red star =
  navigation goal

  You can open the video with any player (VLC, mpv, etc.):
  # after the run finishes:
  ls -t
  /home/nitish/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/results/*.mp4
  | head -1 | xargs vlc

  Or to watch it inline if you're in a Jupyter environment, just open the file
  path the script prints at the end.

✻ Crunched for 14s

※ recap: Building an Active Inference robot navigation system in Habitat-Sim. Just implemented Stage 3 wall-following exploration and tuned max_follow to 30 to cover multiple headings. Next: run the demo and check results.

❯ look at the last result that I executed it timed out. from the visualisation  
  it seems that the robot is going out of the room                              

  Searched for 2 patterns, read 1 file, listed 1 directory, ran 1 shell command (ctrl+o to expand)

● I can see the two problems clearly. Let me get the full picture:

  Searched for 3 patterns (ctrl+o to expand)

● The two bugs are clear:

  1. VIA-DOOR locks permanently — doorway target was at (-2.17, 0.24) but the
  agent orbits at y≈1.1–1.5, always ~1.1m away, never reaching
  door_pass_radius=0.8m. The high wall-penalty flanking the doorway slot makes
  the EFE avoid approaching it, so the agent spins forever.
  2. "Going out of the room" — SEEK(N) traveled 3.5m north and FOLLOW(N) found a
  gap in the north wall (a doorway) and passed through into the adjacent room.
  That's the wall-follower working correctly, but it means the agent is spending
  explore steps in the wrong room.

  Three targeted fixes:

  Read 1 file (ctrl+o to expand)
  ⎿  Referenced file ../results/aif_explore_nav_20260624_002631.txt
  ⎿  Referenced file aif_explore_nav_demo.py
  ⎿  Read ../../../../../.claude/projects/-home-nitish-Documents-AIF-code/memory
     /project_aif_v6_explore_nav.md (264 lines)
  ⎿  Referenced file ../results/aif_explore_nav_20260624_001625.txt
  ⎿  Read ../planner/__init__.py (4 lines)

● Let me read the current state of the two files I need to fix.

  Read 2 files (ctrl+o to expand)

● Now let me read the wall follower:

  Read 1 file (ctrl+o to expand)

● I have a clear picture. Three fixes needed:

  1. VIA-DOOR: door_routing_min_dist_m guard — the agent ended exploration ~1.6m
  from the goal, so VIA-DOOR should never have fired. Guard against activation
  when already close.
  2. VIA-DOOR: timeout escape — if routing gets stuck longer than N steps, bail
  out unconditionally.
  3. SEEK distance — reduce max_seek from 15 → 8 (2m cap) so SEEK(N) doesn't
  penetrate into the next room.

● Update(~/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/model_v6/config.p
        y)
  ⎿  Added 4 lines, removed 2 lines
      166      w_wall_penalty:   float = 5.0     # EFE penalty weight
      167      wall_margin_m:    float = 0.8     # Gaussian sigma for wal
           l influence (metres)
      168      # Doorway routing: redirect goal through a doorway slot wh
           en wall blocks path.
      169 -    door_block_margin_m: float = 0.70 # wall within this dist 
          -of path-to-goal = blocked                                     
      170 -    door_pass_radius_m:  float = 0.80 # dist to doorway slot c
          -ounts as "passed through"                                     
      169 +    door_block_margin_m:    float = 0.70  # wall within this d
          +ist of path-to-goal = blocked                                 
      170 +    door_pass_radius_m:     float = 1.20  # dist to doorway sl
          +ot counts as "passed through"                                 
      171 +    door_routing_min_dist_m: float = 2.50 # don't activate VIA
          +-DOOR when this close to goal                                 
      172 +    door_max_routing_steps: int   = 20    # give up VIA-DOOR i
          +f stuck for this many steps                                   
      173  
      174  
      175  @dataclass

● Now update the planner to add _via_door_steps and both guards:

● Update(~/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/planner/efe_plann
        er_v4.py)
  ⎿  Added 3 lines, removed 2 lines
      185          # True final navigation target — never redirected
      186          self._nav_goal:    Optional[torch.Tensor] = None
      187          # Doorway-routing state
      188 -        self._via_door:    bool                   = False     
      189 -        self._door_target: Optional[torch.Tensor] = None      
      188 +        self._via_door:       bool                   = False  
      189 +        self._door_target:    Optional[torch.Tensor] = None   
      190 +        self._via_door_steps: int                    = 0   # s
          +teps spent in VIA-DOOR mode                                   
      191  
      192          if goal_pos is not None:
      193              self.set_goal(goal_pos)

● Update(~/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/planner/efe_plann
        er_v4.py)
  ⎿  Added 6 lines, removed 5 lines
      198  
      199      def set_goal(self, goal_pos: torch.Tensor) -> None:
      200          """Set (or update) the final navigation goal. Resets d
           oor-routing state."""
      201 -        goal_pos          = goal_pos.float()                  
      202 -        self.goal_pos     = goal_pos                          
      203 -        self._nav_goal    = goal_pos.clone()                  
      204 -        self._via_door    = False                             
      205 -        self._door_target = None                              
      201 +        goal_pos             = goal_pos.float()               
      202 +        self.goal_pos        = goal_pos                       
      203 +        self._nav_goal       = goal_pos.clone()               
      204 +        self._via_door       = False                          
      205 +        self._door_target    = None                           
      206 +        self._via_door_steps = 0                              
      207  
      208      @property
      209      def is_via_door(self) -> bool:

● Now replace the door-routing logic in select_action() with both guards:
● Update(~/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/planner/efe_plann
        er_v4.py)
  ⎿  Added 28 lines, removed 17 lines
      266          #
      267          # State transitions:
      268          #   DIRECT → VIA_DOOR  : direct path to _nav_goal cros
           ses a wall slot
      269 -        #                         AND at least one doorway slo
          -t is known.                                                   
      270 -        #   VIA_DOOR → DIRECT  : agent is within door_pass_rad
          -ius of door target                                            
      271 -        #                         (door has been passed throug
          -h).                                                           
      269 +        #                         AND at least one doorway slo
          +t is known                                                    
      270 +        #                         AND we are still far from th
          +e goal (min-dist guard).                                      
      271 +        #   VIA_DOOR → DIRECT  : agent within door_pass_radius
          + of door target                                               
      272 +        #                         OR stuck in VIA-DOOR for too
          + many steps (timeout).                                        
      273          if self._via_door and self._door_target is not None:
      274 +            self._via_door_steps += 1                         
      275              dist_to_door = (
      276                  belief.pose_mu[:2].to(device) - self._door_tar
           get.to(device)
      277              ).norm()
      276 -            if dist_to_door <= self.cfg.door_pass_radius_m:   
      277 -                # Passed through the door — revert to original
          - goal                                                         
      278 -                self._via_door    = False                     
      279 -                self._door_target = None                      
      278 +            # Exit VIA-DOOR: passed through door, OR stuck too
          + long                                                         
      279 +            if (dist_to_door <= self.cfg.door_pass_radius_m   
      280 +                    or self._via_door_steps >= self.cfg.door_m
          +ax_routing_steps):                                            
      281 +                self._via_door       = False                  
      282 +                self._door_target    = None                   
      283 +                self._via_door_steps = 0                      
      284                  if self._nav_goal is not None:
      285                      self.goal_pos = self._nav_goal
      286          elif (
     ...
      288              and not self._via_door
      289              and wall_xy is not None
      290          ):
      287 -            curr_xy = belief.pose_mu[:2].to(device)           
      288 -            if _is_path_blocked(curr_xy, self._nav_goal.to(dev
          -ice),                                                         
      289 -                                wall_xy, self.cfg.door_block_m
          -argin_m):                                                     
      290 -                best_door = _find_best_doorway(               
      291 -                    curr_xy, self._nav_goal.to(device), door_x
          -y                                                             
      292 -                )                                             
      293 -                if best_door is not None:                     
      294 -                    self._via_door    = True                  
      295 -                    self._door_target = best_door.to(device)  
      296 -                    self.goal_pos     = best_door.to(device)  
      291 +            curr_xy  = belief.pose_mu[:2].to(device)          
      292 +            nav_dist = float((curr_xy - self._nav_goal.to(devi
          +ce)).norm().item())                                           
      293 +            # Only activate via-door when still far enough fro
          +m the final goal.                                             
      294 +            # At close range EFE + wall_penalty are sufficient
          +; door-routing                                                
      295 +            # at close range tends to oscillate because the sa
          +me wall slots keep                                            
      296 +            # triggering _is_path_blocked after the agent "pas
          +ses through".                                                 
      297 +            if nav_dist > self.cfg.door_routing_min_dist_m:   
      298 +                if _is_path_blocked(curr_xy, self._nav_goal.to
          +(device),                                                     
      299 +                                    wall_xy, self.cfg.door_blo
          +ck_margin_m):                                                 
      300 +                    best_door = _find_best_doorway(           
      301 +                        curr_xy, self._nav_goal.to(device), do
          +or_xy                                                         
      302 +                    )                                         
      303 +                    if best_door is not None:                 
      304 +                        self._via_door       = True           
      305 +                        self._door_target    = best_door.to(de
          +vice)                                                         
      306 +                        self.goal_pos        = best_door.to(de
          +vice)                                                         
      307 +                        self._via_door_steps = 0              
      308  
      309          # ---- Batched imagination rollout ----
      310          roll = self.model.imagine_rollout(belief, candidates)

● Now fix the max_seek in the demo and raise the wall-penalty log threshold:

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

● Update(aif_explore_nav_demo.py)
  ⎿  Added 1 line, removed 1 line
      316      # ── wall-following explorer (Stage 3) ───────────────────
           ──────────────────
      317      wall_follower = WallFollowExplorer(
      318          turn_step   = cfg.pose.turn_step_rad,
      319 -        max_seek    = 15,   # forward steps per heading before
          - giving up                                                    
      319 +        max_seek    = 8,    # forward steps per heading (8 × 0
          +.25m = 2m max — stays within room)                            
      320          max_follow  = 30,   # max steps per wall-follow segmen
           t (allows ~2-3 headings in 80 steps)
      321          return_dist = 0.8,  # metres — "returned to start" thr
           eshold
      322      )

● Update(aif_explore_nav_demo.py)
  ⎿  Added 1 line, removed 1 line
      556          log(f"{step+1:>4} | {phase:>8} | {ACTION_NAMES[action]
           :>7} | "
      557              f"({x:6.2f}, {y:6.2f}, {math.degrees(th):5.1f}°) |
            "
      558              f"{goal_dist:8.3f} | {n_slots:>5} | {ms:5.0f}{door
           _tag}")
      559 -        if wall_pen_val > 1.0:                                
      559 +        if wall_pen_val > 5.0:                                
      560              log(f"       [wall-penalty={wall_pen_val:.2f}]")
      561  
      562          # ── render frame ────────────────────────────────────
           ──────────────────

● Let me do a quick sanity check on the critical planner section:

  Read 1 file (ctrl+o to expand)

● All three changes are in place. Here's what was fixed and why:

  ---
  Fix 1 — VIA-DOOR min-distance guard (door_routing_min_dist_m = 2.5m):
  The failing run ended exploration ~1.6m from the nav goal. VIA-DOOR should
  never fire at that range — EFE + wall penalty are sufficient. The guard
  prevents activation when nav_goal_distance ≤ 2.5m.

  Fix 2 — VIA-DOOR timeout (door_max_routing_steps = 20):
  Safety net for legitimate long-range door-routing. If the agent is in VIA-DOOR
  mode for 20 steps without reaching the doorway slot (which can be
  misestimated), it gives up and reverts to direct navigation. This caps the
  worst-case damage at 20 wasted steps instead of 108.

  Fix 3 — door_pass_radius_m 0.8 → 1.2m:
  The doorway slot position is estimated from depth images and may be off by
  0.3–0.4m. Enlarging the "passed through" radius makes successful exit more
  forgiving of imprecise doorway estimates.

  Fix 4 — max_seek 15 → 8 (2m cap):
  SEEK(N) with max_seek=15 could travel 3.75m north, crossing through a doorway
  into the next room. Capping at 8 × 0.25m = 2.0m keeps the probe within the
  starting room for typical apartment rooms. If no wall is found in 2m, that
  heading is skipped and the follower moves on.

  Fix 5 — wall-penalty log threshold 1.0 → 5.0:
  The penalty is nearly always in the 2–13 range, so the old threshold spammed
  every step. Raising to 5.0 only logs genuinely large penalties.

  Run the same command to test:
  conda activate habitat
  cd .../Symbolic_AIF_v6/habitat_adapter
  python aif_explore_nav_demo.py

  You should now see: no VIA-DOOR activation when the agent ends exploration
  close to the goal, and exploration staying within the starting room.

✻ Cooked for 8m 42s