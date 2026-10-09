import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from src.utils import get_mario_state_at_time, get_gc_state_at_time


def choose_ball_position(mario_state, gc_state, last_gc_ball, last_mario_ball, current_time, max_ball_age, max_ball_jump):
    
    # TODO (?) ricorda alla fine elimina tutte le righe None

    if last_gc_ball is not None and (last_mario_ball is None or last_gc_ball[2] > last_mario_ball[2]):
        last_ball = last_gc_ball
    elif last_mario_ball is not None:
        last_ball = last_mario_ball
    else:
        last_ball = None

    if last_ball is not None:
        last_ball_xy = np.array(last_ball[:2])
   

    # 1. Se MARIO vede la palla
    #mario_balls = mario_state[mario_state.id == -1]
    mario_balls = mario_state[mario_state.type == 'ball']
    if current_time < 1000:
       print(f"[DEBUG] t={current_time} | mario_balls len: {len(mario_balls)}")
    # se in questo momento nel MARIO-video viene rilevata una sola palla quella è la MARIO-palla per quel momento
    if len(mario_balls) == 1:
        ball_x, ball_y = mario_balls.iloc[0].field_x, mario_balls.iloc[0].field_y
        mario_ball_time = mario_balls.iloc[0].gametime
                # **NUOVO: Controllo jump per palla MARIO**
        if last_mario_ball is not None:
            last_ball_pos = np.array(last_mario_ball[:2])
            current_ball_pos = np.array([ball_x, ball_y])
            jump_distance = np.linalg.norm(current_ball_pos - last_ball_pos)
            
            # Debug per il controllo jump
            if 54000 <= current_time <= 55000:  # O il range che ti interessa
                print(f"\n[DEBUG MARIO BALL JUMP] t={current_time:.0f}ms")
                print(f"  Palla MARIO corrente: ({ball_x:.1f}, {ball_y:.1f})")
                print(f"  Ultima palla MARIO: ({last_ball_pos[0]:.1f}, {last_ball_pos[1]:.1f})")
                print(f"  Jump distance: {jump_distance:.1f}mm")
                print(f"  max_ball_jump: {max_ball_jump}mm")
            
            if jump_distance > max_ball_jump:
                if 54000 <= current_time <= 55000:
                    print(f"  -> MARIO BALL JUMP TROPPO GRANDE! Usando logica GC...")
                # Jump troppo grande, ignora la palla MARIO e usa la logica GC
                pass  # Continua con la logica GC qui sotto
            else:
                if 54000 <= current_time <= 55000:
                    print(f"  -> MARIO ball jump OK")
                return ball_x, ball_y, 'mario_singleball'
        else:
            # Prima palla MARIO, accettala sempre
            return ball_x, ball_y, 'mario_singleball'
        

        #if current_time < 10000:
         #   print(f"[DEBUG MARIO BALL] t={current_time:.0f} ms | mario_ball_time={mario_ball_time}")
        #return ball_x, ball_y, 'mario_singleball'
    elif len(mario_balls) > 1:
        # **NUOVO: Controllo jump anche per multiple balls**
        valid_mario_balls = []
        
        for _, mario_ball in mario_balls.iterrows():
            ball_pos = np.array([mario_ball.field_x, mario_ball.field_y])
            
            # Calcola jump solo se c'è una ultima palla MARIO
            if last_mario_ball is not None:
                jump_distance = np.linalg.norm(ball_pos - np.array(last_mario_ball[:2]))
                if jump_distance <= max_ball_jump:
                    valid_mario_balls.append((mario_ball, jump_distance))
            else:
                # Prima volta, accetta tutte le palle
                valid_mario_balls.append((mario_ball, 0))
        
        if valid_mario_balls:
            # **Se ci sono palle valide, usa la logica esistente ma solo su quelle**
            if last_gc_ball is not None:
                # Scegli quella più vicina alla GC tra le palle con jump OK
                best_mario_ball = None
                best_dist_to_gc = float('inf')
                
                for mario_ball, jump_dist in valid_mario_balls:
                    ball_pos = np.array([mario_ball.field_x, mario_ball.field_y])
                    dist_to_gc = np.linalg.norm(ball_pos - np.array(last_gc_ball[:2]))
                    if dist_to_gc < best_dist_to_gc:
                        best_dist_to_gc = dist_to_gc
                        best_mario_ball = mario_ball
                
                ball_x, ball_y = best_mario_ball.field_x, best_mario_ball.field_y
                return ball_x, ball_y, 'mario_multiball'
            else:
                # Se non c'è GC, scegli quella con jump più piccolo
                best_mario_ball, _ = min(valid_mario_balls, key=lambda x: x[1])
                ball_x, ball_y = best_mario_ball.field_x, best_mario_ball.field_y
                return ball_x, ball_y, 'mario_multiball_first'
        else:
            # **Tutte le palle MARIO hanno jump troppo grande, continua con logica GC**
            pass
        #elif last ball too old o non c'è mai stata? Se è too old vedere il ballage TODO
    
    # 2. Se MARIO non vede la palla, cerca la GC-palla
    # consideriamo palle gc recenti   
    if mario_balls.empty:  
        gc_balls = gc_state[(gc_state.ballage < max_ball_age) & 
                           (gc_state.ballage > 0) &
                           (gc_state.ballx.notna()) & 
                           (gc_state.bally.notna())]
        if not gc_balls.empty: 
            positions = gc_balls[['ballx', 'bally']].values
             # Robot che vedono la palla dal 54000 al 55000**
            if 54000 <= current_time <= 55000:
                print(f"\n[DEBUG GC BALLS] t={current_time:.0f} ms | Robot con palla: {len(gc_balls)}")
                if last_ball is not None:
                    print(f"  Ultima palla nota: ({last_ball[0]:.1f}, {last_ball[1]:.1f}) al tempo {last_ball[2]:.0f}ms")
                else:
                    print(f"  Nessuna ultima palla nota")
                                    # Stampa dettagli di ogni robot che vede la palla
                for idx, row in gc_balls.iterrows():
                    robot_pos = (row['x'], row['y'])
                    ball_local_pos = (row['ballx'], row['bally'])
                    
                    # Calcola distanza dalla ultima palla nota (se esiste)
                    if last_ball is not None:
                        dist_from_last = np.linalg.norm(np.array(ball_local_pos) - np.array(last_ball[:2]))
                    else:
                        dist_from_last = None
                    
                    print(f"    Robot ({row['team']}, {row['player']}): pos_robot=({robot_pos[0]:.1f}, {robot_pos[1]:.1f}) | ball_local=({ball_local_pos[0]:.1f}, {ball_local_pos[1]:.1f}) | ballage={row['ballage']:.1f}")
                    if dist_from_last is not None:
                        print(f"      -> Distanza da ultima palla: {dist_from_last:.1f}mm")
            # DEBUG: stampa i candidati GC robot tra 30 e 60 secondi
            # if 54000 <= current_time <= 38000:
            #     print(f"[DEBUG GC CANDIDATES] t={current_time} ms | GC robot count: {len(gc_balls)}")
            #     print("GC robot positions:")
            #     for idx, row in gc_balls.iterrows():
            #         print(f"  id={row['player']} team={row['team']} pos=({row['ballx']:.1f}, {row['bally']:.1f}) ballage={row['ballage']}")
            #     if last_ball is not None:
            #         print(f"[DEBUG LAST BALL] last_ball=({last_ball[0]:.1f}, {last_ball[1]:.1f}) | rilevata a t={last_ball[2]/1000:.2f}s | source={last_ball[3] if len(last_ball)>3 else 'N/A'}")
            #     else:
            #         print("[DEBUG LAST BALL] Nessuna last_ball disponibile")    
            if len(positions) > 1:
                BALL_CLUSTER_EPS = 800
                BALL_CLUSTER_MIN_SAMPLES = 2
                # controlla algoritmi clustering per prendere la palla per un robot molto sicuro della posizione, quindi molto vicino (crea un cluster con una sola palla)
                clustering = DBSCAN(eps=BALL_CLUSTER_EPS, min_samples=BALL_CLUSTER_MIN_SAMPLES).fit(positions)
                labels, counts = np.unique(clustering.labels_[clustering.labels_ != -1], return_counts=True)
                # DEBUG: stampa i cluster formati tra 30 e 60 secondi
                # if 30000 <= current_time <= 54000:
                #     print(f"[DEBUG GC CLUSTERS] t={current_time} ms | cluster labels: {labels} | counts: {counts}")
                #     for label in labels:
                #         cluster_positions = positions[clustering.labels_ == label]
                #         print(f"  Cluster {label}: {len(cluster_positions)} robot, mean pos=({cluster_positions.mean(axis=0)[0]:.1f}, {cluster_positions.mean(axis=0)[1]:.1f})")
                # **DEBUG CLUSTERING: Risultati clustering dal 54000 al 55000**
                if 54000 <= current_time <= 55000:
                    print(f"  CLUSTERING: eps={BALL_CLUSTER_EPS}, min_samples={BALL_CLUSTER_MIN_SAMPLES}")
                    print(f"  Cluster labels trovati: {labels}")
                    print(f"  Cluster counts: {counts}")
                    print(f"  Outliers (label -1): {np.sum(clustering.labels_ == -1)}")
                    
                    # Mostra dettagli di ogni cluster
                    for label in np.unique(clustering.labels_):
                        cluster_mask = clustering.labels_ == label
                        cluster_positions = positions[cluster_mask]
                        cluster_robots = gc_balls[cluster_mask]
                        
                        if label == -1:
                            print(f"    OUTLIERS ({len(cluster_positions)} robot):")
                        else:
                            mean_pos = cluster_positions.mean(axis=0)
                            print(f"    CLUSTER {label} ({len(cluster_positions)} robot): mean=({mean_pos[0]:.1f}, {mean_pos[1]:.1f})")
                        
                        for idx, (_, robot) in enumerate(cluster_robots.iterrows()):
                            pos = cluster_positions[idx]
                            print(f"      Robot ({robot['team']}, {robot['player']}): ball=({pos[0]:.1f}, {pos[1]:.1f})")
                    
                    print(f"  -> Cluster utilizzabile trovato: {len(labels) > 0}")

                if len(labels) > 0:
                    # Scegli il cluster più numeroso
                    best_label = labels[np.argmax(counts)]
                    cluster_positions = positions[clustering.labels_ == best_label]

                    ball_x, ball_y = cluster_positions.mean(axis=0)
                    if 54000 <= current_time <= 55000:
                        print(f"  -> CLUSTER SCELTO: {best_label} con {counts[np.argmax(counts)]} robot")
                        print(f"  -> Posizione finale palla: ({ball_x:.1f}, {ball_y:.1f})")
                    
                    # --- CONTROLLO LIMITE SPOSTAMENTO PALLA ---
                    jump = np.linalg.norm(np.array([ball_x, ball_y]) - last_ball_xy)
                    if 54000 <= current_time <= 55000:
                            print(f"  -> Jump check: distanza={jump:.1f}mm, max_allowed={max_ball_jump}mm")
                    if jump > max_ball_jump:
                        if 54000 <= current_time <= 55000:
                            print(f"  -> JUMP TROPPO GRANDE! Usando ultima posizione nota.")
                        return float(last_ball[0]), float(last_ball[1]), 'not_found'
                    return ball_x, ball_y, 'gc_cluster'
                else:   
                    if 54000 <= current_time <= 55000:
                        print(f"  -> NESSUN CLUSTER VALIDO! Tentando fallback...")  
                    # Se non si formano cluster, #### fallback: GC-palla più vicina all'ultima MARIO-palla
                    # Usa come riferimento l'ultima palla scelta (MARIO o GC)

                    if last_ball is not None:
                        # Calcola la distanza tra ogni robot GC e l'ultima palla scelta
                        ROBOT_LASTBALL_MAX_DIST = 2000  # mm (2 metri, modifica a piacere)
                        robot_positions = gc_balls[['x', 'y']].values
                        
                        robot_lastball_dists = np.linalg.norm(robot_positions - last_ball_xy, axis=1)

                        # Filtro: solo robot abbastanza vicini all'ultima palla scelta
                        valid_idx = np.where(robot_lastball_dists < ROBOT_LASTBALL_MAX_DIST)[0]
                        if len(valid_idx) > 0:
                            filtered_positions = positions[valid_idx]
                            filtered_gc_balls = gc_balls.iloc[valid_idx]
                            filtered_dists = np.linalg.norm(filtered_positions - last_ball_xy, axis=1)
                            idx = np.argmin(filtered_dists)
                            ball_x, ball_y = filtered_positions[idx]

                            # --- CONTROLLO LIMITE SPOSTAMENTO PALLA ---
                            jump = np.linalg.norm(np.array([ball_x, ball_y]) - last_ball_xy)
                            if jump > max_ball_jump:       
                                return float(last_ball[0]), float(last_ball[1]), 'not_found'
                            return ball_x, ball_y, 'gc_nearest_to_last_ball_filtered'

                        else:
                            return float(last_ball[0]), float(last_ball[1]), 'not_found'

    # 3. Se non c'è una GC-palla, usa l'ultima GC-palla o l'ultima MARIO-palla se non troppo vecchie
    # se 
    #if last_gc_ball is not None and (current_time - last_gc_ball[2]) < max_ball_age:
    #    return last_gc_ball[0], last_gc_ball[1], 'last_gc'
    # se non c'è una mario palla e non c'è una gc palla prendi mario palla recente se c'è    
    if last_mario_ball is not None and (current_time - last_mario_ball[2]) < max_ball_age:
        return last_mario_ball[0], last_mario_ball[1], 'mario_last'
    # print("non rientra nei casi")
    if last_ball is None:
        return None, None, 'not_found'
    else:
        return float(last_ball[0]), float(last_ball[1]), 'not_found'

def kalman_smooth_ball(ball_df, process_var=1.0, measure_var=50.0):
    """
    Applica un filtro di Kalman 2D semplice alle posizioni della palla.
    """
    x = ball_df['chosen_ball_x'].values
    y = ball_df['chosen_ball_y'].values

    n = len(x)
    smooth_x = np.zeros(n)
    smooth_y = np.zeros(n)

    # Stato iniziale: posizione iniziale
    smooth_x[0] = x[0]
    smooth_y[0] = y[0]
    # Incertezza iniziale
    p_x = 1.0
    p_y = 1.0

    for i in range(1, n):
        # Predizione: stato precedente
        pred_x = smooth_x[i-1]
        pred_y = smooth_y[i-1]
        p_x += process_var
        p_y += process_var

        # Aggiornamento: misura corrente
        k_x = p_x / (p_x + measure_var)
        k_y = p_y / (p_y + measure_var)
        smooth_x[i] = pred_x + k_x * (x[i] - pred_x)
        smooth_y[i] = pred_y + k_y * (y[i] - pred_y)
        p_x = (1 - k_x) * p_x
        p_y = (1 - k_y) * p_y

    ball_df['kalman_x'] = smooth_x
    ball_df['kalman_y'] = smooth_y
    return ball_df

def build_ball_dataset(mario_df, gc_fixed_penalties, initial_ball, ball_source, s_config):
    ball_rows = []
    timestamps = sorted(mario_df['gametime'].unique())
    last_mario_ball = None
    
    # Inizializzazione memoria palla
    if ball_source and ball_source.startswith('gc'):
        last_gc_ball = initial_ball
        last_mario_ball = None
    elif ball_source and ball_source.startswith('mario'):
        last_mario_ball = initial_ball
        last_gc_ball = None
    else:
        last_gc_ball = None
        last_mario_ball = None

    for i, timestamp in enumerate(timestamps):
        if timestamp < 0:
            continue
        if i % 1000 == 0 or i == len(timestamps) - 1:
            print(f"[BALL DATASET] Frame {i+1}/{len(timestamps)} (gametime: {timestamp:.0f} ms)")
            
        mario_state = get_mario_state_at_time(mario_df, timestamp)
        gc_state = get_gc_state_at_time(gc_fixed_penalties, timestamp)

        # **USA SOLO QUESTA CHIAMATA - NON DUPLICARE**
        ball_x, ball_y, source = choose_ball_position(
            mario_state,
            gc_state,
            last_gc_ball,
            last_mario_ball,
            timestamp,
            s_config.max_ball_age,
            s_config.max_ball_jump,
        )
        
        # Aggiorna la memoria delle palle
        if source and source.startswith('gc'):
            last_gc_ball = (ball_x, ball_y, timestamp, source)
        elif source and source.startswith('mario'):
            last_mario_ball = (ball_x, ball_y, timestamp, source)
        elif source and source == 'not_found':
            last_mario_ball = (ball_x, ball_y, timestamp, source)
            last_gc_ball = (ball_x, ball_y, timestamp, source)

        # Calcola le palle "grezze" per il dataset
        gc_balls = gc_state[(gc_state.ballage > 0) & (gc_state.ballage < s_config.max_ball_age)]
        is_playing = gc_state.playing.any() if not gc_state.empty else False
        
        # Inizializza le variabili
        mario_ball_x = mario_ball_y = gc_ball_x = gc_ball_y = None
        
        if source == "not_found":
            pass  # Tutte già None
        elif source in ["gc_cluster", "gc_nearest_to_last_ball_filtered"]:
            gc_ball_x = ball_x
            gc_ball_y = ball_y
        elif source in ["mario_singleball", "mario_multiball", "mario_multiball_first", "mario_last"]:
            mario_ball_x = ball_x
            mario_ball_y = ball_y
            if not gc_balls.empty:
                positions = gc_balls[['ballx', 'bally']].values
                if len(positions) > 1:
                    clustering = DBSCAN(eps=100, min_samples=2).fit(positions)
                    labels, counts = np.unique(clustering.labels_[clustering.labels_ != -1], return_counts=True)
                    if len(labels) > 0:
                        best_label = labels[np.argmax(counts)]
                        cluster_positions = positions[clustering.labels_ == best_label]
                        gc_ball_x, gc_ball_y = cluster_positions.mean(axis=0)
        
        if timestamp <= 62000:
            print(f"[DEBUG BALL GLOBAL] t={timestamp:.0f} ms | pos=({ball_x}, {ball_y}) | source={source}")
            
        ball_rows.append({
            'timestamp_ms': timestamp,
            'mario_ball_x': mario_ball_x,
            'mario_ball_y': mario_ball_y,
            'gc_ball_x': gc_ball_x,
            'gc_ball_y': gc_ball_y,
            'chosen_ball_x': ball_x,
            'chosen_ball_y': ball_y,
            'ball_source': source,
            'is_playing': is_playing
        })
    return pd.DataFrame(ball_rows)