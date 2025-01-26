import argparse, sys
import os
import pathlib
os.chdir((pathlib.Path(__file__) / ".." / ".." / "..").resolve())
# print (os.getcwd())
import sys
sys.path.append("./")

import shared_config




parser = argparse.ArgumentParser()

parser.add_argument("--starttime", help="Incident Start time in seconds")
parser.add_argument("--endtime", help="Incident End time in seconds")
parser.add_argument("--which_type_of_cells", help="Type of cells to print using fortran")

cl_args = parser.parse_args()

frozen_part_fortran_part_1 = """
PROGRAM CTMSIM
	!	USE MSFLIB
	!	USE PORTLIB
	!----------------
	! DECLARE VARIABLES
	!----------------
	IMPLICIT NONE
	REAL V(6400)						! Free flow speed (kph) Array(flow)
	REAL W(6400)						! Shockwave speed (kph)	Array(flow)
	REAL Kj					    ! Jam Density (veh/km-lane)
	REAL SatFlow(6400)				! per lane saturation flow (vph)
	REAL TotVol1(500),TotVol2(500),Vol1,Vol2 ! Cumulative volume at each 15 minute
	INTEGER DT					! Length of time step (sec)
	INTEGER EndSim				! Actual time step of simulation
	INTEGER NumCell 			! Number of cells in the network

	CHARACTER Name(6400)*11		! Name of the cells Array(i)
	REAL FULLfactor(6400)		! Proportion of cell that is full at t=0
	REAL Lanes(6400)			    ! Number of lanes _leaving_ cell(i)
	!		Origin cells should have a number of lanes
	!		Destination should have zero lanes.
	REAL TURNfactor(6400)		! Turning Factor, adjust SatFlow
	!		Qmax(I)= SatFlow*Lanes(I)*TURNfactor(I)
	!		If cell is start of diverge, the program calculates
	!		that cell's Qmax by looking at the Qmaxs of the downstream
	!		branches, overwriting whatever initial value the cell has
	CHARACTER CellType(6400)*2	! Type of Cell
	!		OR=	Origin				Create holding capacity of 99,999
	!		TR=	Transition			Not examined
	!		DE=	Destination			Create holding cap of 99,999, set delay array=0
	!		DI=	Start of Diverge	Split cell's outflow
	!		DB= Diverge Branch	    Not examined
	!		ME=	(UnSig)Merge Branch Inflow of downstream cell is sum of MB cells
	!		MB=	(Sig) Merge Branch	Inflow of downstream cell is sum of MB cells
	CHARACTER NameFrom1(6400)*11	! Name of First 'from' link
	CHARACTER NameFrom2(6400)*11	! Name of Second 'from' link
	CHARACTER NameTo1(6400)*11	! Name of First 'to' link
	CHARACTER NameTo2(6400)*11	! Name of Second 'to' link
	INTEGER From1(6400)			! First 'from' link
	INTEGER From2(6400)			! Second 'from' link
	INTEGER To1(6400)			! First 'to' link
	INTEGER To2(6400)			! Second 'to' link
	REAL Split1(6400)			! Proportion to first branch
	! Currently: For head of diverge only
	REAL Split2(6400)			! Proportion to second branch
	REAL Priority(6400)			! Priority of the unsignalize merge
	CHARACTER SigQuery(6400)		! Signal regulating outflow? ('Y' or 'y'=yes)
	INTEGER Offset(6400)			! Signal offset
	INTEGER Greff(6400)			! Effecetive green duration
	INTEGER Redeff(6400)			! Effecetive green duration
	INTEGER CYC					! Cycle length
	INTEGER IntoCycle           ! Time into cycle -phase at T
	INTEGER T					! time step index
	INTEGER I					! cell number index
	REAL Q						! Max cell flow
	INTEGER NumLoad				! Number of loadings
	CHARACTER NameLoadCell*11	! Name of Cell being loaded
	INTEGER LoadCell(50)			! Cell being loaded
	REAL Dem1(72000),Dem2(72000),Dem3(72000) ! Demand into cell (veh/hour)
	REAL Dem4(72000),Dem5(72000),Dem6(72000)
	REAL Dem7(72000),Dem8(72000),Dem9(72000)
	REAL Dem10(72000),Dem11(72000),Dem12(72000)
	REAL Dem13(72000),Dem14(72000),Dem15(72000)
	REAL Dem16(72000),Dem17(72000),Dem18(72000)
	REAL Dem19(72000),Dem20(72000),Dem21(72000)
	REAL Dem22(72000),Dem23(72000),Dem24(72000)
	REAL Dem25(72000),Dem26(72000),Dem27(72000)
	REAL Dem28(72000),Dem29(72000),Dem30(72000)
	REAL Dem31(72000),Dem32(72000),Dem33(72000)
	REAL Dem34(72000),Dem35(72000),Dem36(72000)
	REAL Dem37(72000),Dem38(72000),Dem39(72000)
	REAL Dem40(72000),Dem41(72000),Dem42(72000)
	REAL Dem43(72000),Dem44(72000),Dem45(72000)
	REAL Dem46(72000),Dem47(72000),Dem48(72000)
	REAL Dem49(72000),Dem50(72000),Dem51(72000)
	
	INTEGER LoadSt				! Start time step of loading
	INTEGER LoadEnd				! End time step of loading
	INTEGER b,J,K
	INTEGER NN					!Number of 15 minute intervals
	INTEGER Count				! Generic counting variable
	REAL Send
	REAL Recieve
	INTEGER OppNo
	
	
	!====ACCIDENT PARAMETERS
	! INTEGER Acc_start  ! In seconds
	! INTEGER Acc_end ! In second, large than Acc_start
	! REAL CapRedu (72000)  !Same size as Name, the remianing capacity proportion, e.g. 60% capacity left after accident then it is 0.6
	! INTEGER AccidentCellName !Single cell for accidents
    
    INTEGER NumAccidents, ii, jj, kk
    INTEGER AccidentCellNames(100), AccStart(100), AccEnd(100)
    REAL CapReduNew(100)
	!===================================================

	

	REAL MaxFactor(6400)
	REAL HoldCap (6400)			! Cell holding capacity array
	!		NOTE:  If the cell is an origin or destination, the program
	!		assigns a holing capacity of 99,999 vehicles.
	REAL HoldCapPreAccident (6400)			! Backup Cell holding capacity to restore flow after accidents  
	REAL Qmax(6400)				! Max outflow from cell(i)
	REAL QmaxBackupPreAccident(6400)				! Backup Cell holding QMax to restore flow after accidents
	REAL N(6400,0:72000)			! Cell occupancy array (i,t)
	REAL S(6400,0:72000)			! Cell outflows (i,t)
	REAL Y(6400,0:72000),des1(0:72000),des2(0:72000)			! Cell inflows array (i,t)
	COMMON/BLOCK1/V,W,Split1,Split2,HoldCap,Qmax,N,S,Y,&
			Priority,DT,EndSim,NumCell,From1,From2,To1,To2


	! ------------------------------
	!	TITLE BAR
	! ------------------------------
	PRINT*,'   ** CTM SIMULATION FOR OSM STREETS **'

	! ------------------------------
	!	Read and Write Global Parameters
	! ------------------------------
	PRINT*,'Running FORTRAN Code'
	OPEN (2,FILE='input_output_text_files/input.txt')	! Scenario File
	OPEN (3,FILE='input_output_text_files/out30.txt')
    ! OPEN (4, FILE='input_output_text_files/accident.txt')
	OPEN (6,FILE='input_output_text_files/outfl.txt')	! Output File
	PRINT*,'Reading and Writing Global Parameters...'


	READ (2,*) NumCell
	READ (2,*) DT,EndSim
	READ (2,*) Kj
	PRINT*,'@Nishant read first three lines'
	WRITE (6,900)
	900	FORMAT ('********* G L O B A L   P A R A M E T E R S *********')
	WRITE (6,999)
	999 FORMAT (' ')
	WRITE (6,907) Kj
	WRITE (6,909) DT
	WRITE (6,910) EndSim
	907 FORMAT ('Jam Density, Kj (veh/km):',2X,F9.2)
	909 FORMAT ('Time Step (sec):',12X,I5)
	910 FORMAT ('End Simulation Step:',8X,I5)

	! ------------------------------
	! Read and write Cell Information
	! ------------------------------
	WRITE (6,999)
	WRITE (6,999)
	WRITE (6,940)
	940	FORMAT ('********* C E L L   I N F O R M A T I O N *********')
	WRITE (6,999)
	PRINT*,'Reading and Writing Cell Information...    ',NumCell,' Cells'
	WRITE (6,915) NumCell
	WRITE (6,999)
	WRITE (6,977)
	915	FORMAT ('Number of Cells:',I5)
	977 FORMAT ('Entered Cell Data--')
	WRITE (6,999)
	WRITE (6,945)
	945 FORMAT (1X,'No.'3X,'Name',2X,'fFULL',2X,'Lns',3X,'fTURN',3X,'Typ',3X,'Fr1',&
			2X,'Fr2',2X,'To1',2X,'To2',2X,'Spt1',1X,'Spt2',1X,'Prio'&
			2X,'Sig?',2X,'Off.',5X,'Geff',2X,'Reff',2X,'fMax',2X,'fflow',2X,'shkwve',2X,'satflow')
	DO 50 I = 1,NumCell
		READ (2,*) Name(I),FULLfactor(I),Lanes(I),TURNfactor(I),CellType(I),&
				NameFrom1(I),NameFrom2(I),NameTo1(I),NameTo2(I),Split1(I),Split2(I),&
				Priority(I),SigQuery(I),Offset(I),Greff(I),Redeff(I),MaxFactor(I),V(I),W(I),SatFlow(I)
		!		write (*,*) Name(I),FULLfactor(I),Lanes(I),TURNfactor(I),CellType(I),&
		!				NameFrom1(I),NameFrom2(I),NameTo1(I),NameTo2(I),Split1(I),Split2(I),&
		!				Priority(I),SigQuery(I),Offset(I),Greff(I),Redeff(I),MaxFactor(I),V(I),W(I),SatFlow(I)
	50  CONTINUE
	DO 903 I=1,NumCell
		DO 904 b=1,NumCell							! To do the transformation of
			IF (NameFrom1(I) == Name(b)) THEN		! the Name format to the
				From1(I) = b						! corresponding cell#
			ENDIF									! for From1,From2,To1 and To2
			IF (NameFrom2(I) == Name(b)) THEN
				From2(I) = b
			ENDIF
			IF (NameTo1(I) == Name(b)) THEN
				To1(I) = b
			ENDIF
			IF (NameTo2(I) == Name(b)) THEN
				To2(I) = b
			ENDIF
		904 CONTINUE
	903 CONTINUE

	DO 51 I=1,NumCell
		WRITE (6,950) I,Name(I),FULLfactor(I),Lanes(I),TURNfactor(I),CellType(I),&
				From1(I),From2(I),To1(I),To2(I),Split1(I),Split2(I),&
				Priority(I),SigQuery(I),Offset(I),Greff(I),Redeff(I),MaxFactor(I)
		950		FORMAT (I3,3X,A11,F5.3,F6.2,F6.2,4X,A,4I5,3F6.2,4X,A,X,3I5,F6.1,3F10.5)
		!		Create Holding Capacity Array and initial occupancy conditions
		SELECT CASE(CellType(I))
		CASE ('OR','DE','Or','De','or','de','oR','dE')
			!			Create "infinite" holding capacity for origins and destinations
			HoldCap(I)=99999
			N(I,0)=0
		CASE DEFAULT
			HoldCap(I)=(Lanes(I)*V(I)*DT*Kj)/3600
			N(I,0)=FULLfactor(I)*HoldCap(I)
		END SELECT
		!		Create Qmax Array
		Qmax(I)=(SatFlow(I)*TURNfactor(I)*DT*Lanes(I)*MaxFactor(I))/3600
	51	CONTINUE
	PRINT*,'Created Qmax Array'
	PRINT*,'Created Holding Capacity Array'
	PRINT*,'Loaded Initial Conditions'
	!   Load Origin Cells with same number of vehicles as downstream cell.
	!   This will ensure that there is no gap in vehicle flow when
	!   the network is preloaded.
	DO 55 I = 1,NumCell
		SELECT CASE(CellType(I))
		CASE ('OR','Or','or','oR')
			N(I,0)=N(To1(I),0)
		CASE DEFAULT
		END SELECT
	55	CONTINUE

	! -------------------
	! Output Holding Capacity, Qmax Arrays, Initial Cell Occupancies
	! -------------------
	WRITE (6,999)
	WRITE (6,201)
	201	FORMAT ('Calculated Cell Data--')
	DO 202 I=1,NumCell
		WRITE (6,203) I,Qmax(I),HoldCap(I),N(I,0),V(I),Lanes(I), DT, Kj
		203		FORMAT ('Cell:',I5,3X,'Qmax:',F8.2,3X,'Capacity:',F7.2,&
				3X,'Initial n:',F8.2, 'V(I):', F8.2,3X,&
				'Lanes(I):', F8.2,3X, 'DT:', I5,3X, 'Kj:',&
				 F8.2,3X)
	202	CONTINUE
	! ----------------------------
	! Read and Write Demand Information
	! ----------------------------
	WRITE (6,999)
	WRITE (6,999)
	WRITE (6,970)

	970	FORMAT ('********* D E M A N D   I N F O R M A T I O N *********')
	WRITE (6,999)
	READ (2,*) NumLoad
	PRINT*,'Reading and Writing Demand Information...  ',NumLoad,' Loadings'
	WRITE (6,916) NumLoad
	916	FORMAT ('Number of Loadings:',I5)
	WRITE (6,999)
	WRITE (6,975)
	975	FORMAT ('Loading',7X,'Start',6X,'End')
	WRITE (6,980)
	980 FORMAT ('Cell',5X,'Step',7X,'Step')

	DO 60 Count = 1,NumLoad
		READ (2,*) NameLoadCell,LoadSt,LoadEnd
		DO 52 I= 1,NumCell
			IF (NameLoadCell == Name(I)) THEN	! Transformation to the
				LoadCell(Count) = I					! corresponding load cell
			ENDIF
		52		CONTINUE
		WRITE (6,985) LoadCell(Count),LoadSt,LoadEnd
		985		FORMAT (I5,4X,I6,5X,I6)
	60	CONTINUE

	!	@Nishant, commented: because this file is not being populated
	!	But here we are trying to read it, hence the errors
"""

frozen_part_fortran_part_2 = """

	!============accident info====================
	! READ (4,*) Acc_start
	! READ (4,*) Acc_end

    ! No need for this loop, we just need to run once since we simply need to read one cell where accident was  
    ! caused 
	!DO I = 1,NumCell
"""



frozen_part_fortran_part_3 = """
	!END DO
	!==============================================

	! ****************************************
	! Start of CTM Simulation
	! ****************************************
	! -----------------------
	! Determine cell outflows
	! -----------------------
	
	HoldCapPreAccident = HoldCap
    QmaxBackupPreAccident = Qmax
	DO 200 T=0,EndSim,DT


"""

conditional_frozen_part_if_accident = """
		!============= NON-RUCURRENT CONGESTION CREATED BY INTRODUCING ACCIDENT =========
		!===========================NEW ON 3/11/2021 BY YI===============================
		!================================================================================
		! A time point of a day and a road are first selected,e.g., T=? and I=?
		! Then the capacity reduction/availability is sampled from a distribution, e.g., doubly truncated normal distribution
		! The incident duration h is sampled from log-logistic distribution (used for nonmonotonic harzard functions)
		! The hazard function h(t_d) gives the probability density that the incident will be cleared between t_d and t_d+/-dt_d,
		!given that the incident has not yet been cleared up to time t_d, which is the rate of failure, in fact.
		!		  (λP)(λt_d )^(P−1)
		! h(t_d) = -----------------
		!			1 + (λt_d )P
		! λ = 0.0122 and P = 2.212
		! t_d = (P−1)^(1/P)
		!			λ
		!following Ref. [54]. Due to lacking capacity reduction data,
		!the parameters μ and σ in Eq. (18) are arbitrarily assumed to
		!be 0.217 and 0.117.
		!===============================Simple accident based on input file accident.txt=============================
		
		! We save the pre-accident values for Acc_start ( @Acc_start - 1 )
		! So that we can replace it after the accident is over 
		! IF (T==(Accstart-5)) THEN 
		!     DO I=1,NumCell
        !         HoldCapPreAccident(I) = HoldCap(I)
        !         QmaxBackupPreAccident(I) = QMax(I)
        !     END DO 
		! END IF
		
	    ! Applying capacity reductions in the simulation loop; Old version
		!IF ((T>=Acc_start).AND.(T<=Acc_end)) THEN
		! 	DO I=1,NumCell
        !        IF (I==AccidentCellName) THEN 
        !            HoldCap(I)=HoldCapPreAccident(I) * CapRedu(T)
        !            Qmax(I)=QmaxBackupPreAccident(I) * CapRedu(T)
        !        END IF
	    !  	END DO
		!END IF
		!IF (T>Acc_end) THEN
		!	DO I=1,NumCell
		!	    HoldCap(I) = HoldCapPreAccident(I)
		!		Qmax(I)=QmaxBackupPreAccident(I)
		!	END DO
		!END IF
		
		
        ! Before the simulation loop, initialize the backup arrays
        ! ALLOCATE(HoldCapPreAccident(6400), QmaxBackupPreAccident(6400))

        
        ! Simulation loop with accident management

        ! Print total simulation time and number of accidents only once
        ! PRINT*, 'Total simulation time: ', EndSim
        ! PRINT*, 'Number of accidents: ', NumAccidents
        
        ! Simulation loop for time steps
        kk = T
        ! DO 995 kk = 0, EndSim, DT
            DO 998 jj = 1, NumAccidents
                IF (kk == (AccStart(jj) - 5)) THEN
                    HoldCapPreAccident(AccidentCellNames(jj)) = HoldCap(AccidentCellNames(jj))
                    QmaxBackupPreAccident(AccidentCellNames(jj)) = Qmax(AccidentCellNames(jj))
                END IF
                IF (kk >= AccStart(jj) .AND. kk <= AccEnd(jj)) THEN
                    HoldCap(AccidentCellNames(jj)) = HoldCapPreAccident(AccidentCellNames(jj)) * CapReduNew(jj)
                    Qmax(AccidentCellNames(jj)) = QmaxBackupPreAccident(AccidentCellNames(jj)) * CapReduNew(jj)
                ELSE IF (kk > AccEnd(jj)) THEN
                    HoldCap(AccidentCellNames(jj)) = HoldCapPreAccident(AccidentCellNames(jj))
                    Qmax(AccidentCellNames(jj)) = QmaxBackupPreAccident(AccidentCellNames(jj))
                END IF
            998 CONTINUE ! END DO
        ! 995 CONTINUE ! END DO




"""
frozen_part_fortran_part_4  = """
		
		
		
		!=============================================================================================================

		Out:DO 210 I = 1,NumCell
			SELECT CASE (CellType(I))
			CASE ('DI','Di','di','dI')
				!			If cell is start of diverge, look downstream to
				!			ensure that correct Qmax is used
				Qmax(I)=Qmax(To1(I))+Qmax(To2(I))
			CASE DEFAULT
			END SELECT
			Q=Qmax(I)

			!			If signal exists. and if in red phase, change Q
			SigChk: IF ((SigQuery(I).EQ.'Y').OR.(SigQuery(I).EQ.'y')) THEN
				!Signal exists; determine the cycle length and time into cycle
				CYC = Greff(I)+ Redeff(I)
				Before: IF (T.LT.Offset(I)) THEN
					Intocycle = MOD(T-Offset(I),CYC)+CYC
				ELSE Before
					Intocycle = MOD(T-Offset(I),CYC)
				END IF Before
				!determine phase if red then Q=0
				Redchk: IF(Intocycle.GE.Greff(I)) THEN
					Q=0
				ELSE Redchk
				END IF Redchk
			ELSE Sigchk
			ENDIF SigChk

			!			Check if cell is start of diverge
			SELECT CASE(CellType(I))
			CASE ('DI','Di','di','dI')	! cell is start of diverge
				IF ((Split1(I).EQ.0).OR.(Split2(I).EQ.0)) THEN
					PRINT*,'**** Error is diverge split info'
					STOP
				END IF
				S(I,T) = MIN(N(I,T),Q,&
						Qmax(To1(I))/Split1(I),&
						((W(I)/V(I))*(HoldCap(To1(I))-N(To1(I),T)))/Split1(I),&
						Qmax(To2(I))/Split2(I),&
						((W(I)/V(I))*(HoldCap(To2(I))-N(To2(I),T)))/Split2(I))
			CASE ('ME','Me','me','mE')	! cell is start of unsignalize merge
				IF (From1(To1(I)) == I) THEN
					OppNo = From2(To1(I))
					IF (OppNo > I) THEN
						S(I,T)= MIN(Q,N(I,T))
						Send = MIN(Qmax(OppNo),N(OppNo,T))
						Recieve =  MIN(Qmax(To1(I)),W(I)/V(I)*(HoldCap(To1(I))-N(To1(I),T)))
						IF((S(I,T)+Send) > Recieve) THEN
							IF( (Recieve-Send) > Priority(I)*Recieve ) THEN
								S(I,T) = Recieve-Send
							ELSE IF (S(I,T) > Priority(I)*Recieve) THEN
								S(I,T) = Priority(I)*Recieve
							ENDIF
						ENDIF
						IF (Send > Recieve - S(I,T)) THEN
							S(OppNo,T) = Recieve - S(I,T)
						ELSE
							S(OppNo,T) = Send
						ENDIF
					ENDIF
				ELSE IF (From1(To1(I)) > I) THEN
					OppNo = From1(To1(I))
					S(I,T)= MIN(Q,N(I,T))
					Send =	MIN(Qmax(OppNo),N(OppNo,T))
					Recieve =  MIN(Qmax(To1(I)),W(I)/V(I)*(HoldCap(To1(I))-N(To1(I),T)))
					IF((S(I,T)+Send) > Recieve) THEN
						IF( (Recieve-Send) > Priority(I)*Recieve ) THEN
							S(I,T) = Recieve-Send
						ELSE IF (S(I,T) > Priority(I)*Recieve) THEN
							S(I,T) = Priority(I)*Recieve
						ENDIF
					ENDIF
					IF (Send > Recieve - S(I,T)) THEN
						S(OppNo,T) = Recieve - S(I,T)
					ELSE
						S(OppNo,T) = Send
					ENDIF
				ENDIF
			CASE DEFAULT				! cell is not start of a diverge
				IF ((From2(To1(I)) /= 0) .and. (SigQuery(I) == 'N')) THEN
					IF (From1(To1(I)) == I) THEN
						OppNo = From2(To1(I))
						IF (OppNo > I) THEN
							S(I,T)= MIN(Q,N(I,T))
							Send = MIN(Qmax(OppNo),N(OppNo,T))
							Recieve =  MIN(Qmax(To1(I)),W(I)/V(I)*(HoldCap(To1(I))-N(To1(I),T)))
							IF((S(I,T)+Send) > Recieve) THEN
								IF( (Recieve-Send) > Priority(I)*Recieve ) THEN
									S(I,T) = Recieve-Send
								ELSE IF (S(I,T) > Priority(I)*Recieve) THEN
									S(I,T) = Priority(I)*Recieve
								ENDIF
							ENDIF
							IF (Send > Recieve - S(I,T)) THEN
								S(OppNo,T) = Recieve - S(I,T)
							ELSE
								S(OppNo,T) = Send
							ENDIF
						ENDIF
					ELSE IF (From1(To1(I)) > I) THEN
						OppNo = From1(To1(I))
						S(I,T)= MIN(Q,N(I,T))
						Send =	MIN(Qmax(OppNo),N(OppNo,T))
						Recieve =  MIN(Qmax(To1(I)),W(I)/V(I)*(HoldCap(To1(I))-N(To1(I),T)))
						IF((S(I,T)+Send) > Recieve) THEN
							IF( (Recieve-Send) > Priority(I)*Recieve ) THEN
								S(I,T) = Recieve-Send
							ELSE IF (S(I,T) > Priority(I)*Recieve) THEN
								S(I,T) = Priority(I)*Recieve
							ENDIF
						ENDIF
						IF (Send > Recieve - S(I,T)) THEN
							S(OppNo,T) = Recieve - S(I,T)
						ELSE
							S(OppNo,T) = Send
						ENDIF
					ENDIF
				ELSE IF(Split1(I) /= 0) THEN
					S(I,T) = MIN(N(I,T),Q,Qmax(To1(I))/Split1(I),&
							((W(I)/V(I))*(HoldCap(To1(I))-N(To1(I),T)))/Split1(I),&
							Qmax(To2(I))/Split2(I),&
							((W(I)/V(I))*(HoldCap(To2(I))-N(To2(I),T)))/Split2(I))
				ELSE
					S(I,T) = MIN(N(I,T),Q,&
							(W(I)/V(I))*(HoldCap(To1(I))-N(To1(I),T)))
				ENDIF
			END SELECT
			!		210	CONTINUE
		210 END DO Out
		!	NOTE:
		!   The start of a diverge can also have a signal controling its outflow

		! --------------------
		! Determine cell inflow(s)
		! based on sending array, S
		! --------------------
		DO 230 I = 1,NumCell
			SELECT CASE(CellType(I))
			CASE('DI','Di','di','dI')	! Diverge start considerations
				Y(To1(I),T)=S(I,T)*Split1(I)
				Y(To2(I),T)=S(I,T)*Split2(I)
			CASE('MB','Mb','mb','mB',&
					'ME','Me','me','mE')   ! Merge Branch considerations
				Y(To1(I),T)=S(From1(To1(I)),T)+S(From2(To1(I)),T)
			CASE DEFAULT
				IF (From2(To1(I)) /= 0) THEN
					Y(To1(I),T)=S(From1(To1(I)),T)+S(From2(To1(I)),T)
				ELSE IF (Split1(I) /= 0) THEN
					Y(To1(I),T)=S(I,T)*Split1(I)
					Y(To2(I),T)=S(I,T)*Split2(I)
				ELSE
					Y(To1(I),T)=S(I,T)
				ENDIF
			END SELECT
		230		CONTINUE	! Cell inflow loop
		! --------------------
		! Update cell occupancies
		! --------------------
		DO 240 I = 1,NumCell
			SELECT CASE(CellType(I))
			CASE('DE','De','de','dE')	! Sink Considerations
				N(I,T+DT)=N(I,T)+Y(I,T)
			CASE DEFAULT
				N(I,T+DT)=N(I,T)+Y(I,T)-S(I,T)
			END SELECT
		240		CONTINUE
	200	CONTINUE  ! Move on to next time step

"""



which_type_of_cells = cl_args.which_type_of_cells


count_origins = 0
with open("input_output_text_files/input.txt") as f:
    for row in f:
        listed = row.strip().split("	")
        if len(listed) > 10:  # @Nishant: quickfix to filter out the input rows
            if listed[4] == "OR":
                count_origins += 1


print(frozen_part_fortran_part_1)

"""

    DO 70 T = LoadSt,LoadEnd
        READ(3,*)Dem1(T),Dem2(T),Dem3(T),Dem4(T),Dem5(T),Dem6(T)
        Y(LoadCell(1),T)=Dem1(T)*DT
        Y(LoadCell(2),T)=Dem2(T)*DT
        Y(LoadCell(3),T)=Dem3(T)*DT
        Y(LoadCell(4),T)=Dem4(T)*DT
        Y(LoadCell(5),T)=Dem5(T)*DT
        Y(LoadCell(6),T)=Dem6(T)*DT

    70	CONTINUE
"""
print("! reading demand file out30.txt ")
print("        DO 70 T = LoadSt,LoadEnd")
print("              READ(3,*) ", end="")
for i in range(count_origins):
    if i < count_origins - 1:
        print("Dem" + str(i + 1) + "(T),", end="")
    else:
        print("Dem" + str(i + 1) + "(T)", end="\n")  # no comma in the last one

for i in range(count_origins):
    print("                  Y(LoadCell(" + str(i + 1) + "),T) = Dem" + str(i + 1) + "(T)*DT")

print("70	CONTINUE")

print(frozen_part_fortran_part_2)

if not (int(cl_args.starttime) == int(cl_args.endtime)):
    # implies accidents; only then we need the codes for accidents
    # Flexible accidents flow
    # READ(4, *) Name(I), CapRedu(I,1), CapRedu(I,2) ....

    # load_capredu = "READ(4, *) AccidentCellName"
    # for i in range(int(cl_args.starttime), int(cl_args.endtime)):
    #     load_capredu = load_capredu + ", Capredu(" + str(i) + ")"
    # print (load_capredu)

    load_capredu= """
        
        ! Open the accidents file
        OPEN(4, FILE='input_output_text_files/accident.txt')
        
        ! Read the number of accidents
        READ(4,*) NumAccidents
        
        ! Read each accident's details
        DO i = 1, NumAccidents
            READ(4,*) AccidentCellNames(i), AccStart(i), AccEnd(i), CapReduNew(i)
        END DO
    """
    print(load_capredu)


print(frozen_part_fortran_part_3)


if not (int(cl_args.starttime) == int(cl_args.endtime)):
    print(conditional_frozen_part_if_accident)


print (frozen_part_fortran_part_4)


import sys


# Below code is generated using python
"""
	!	Outflow from main street

	WRITE(6,999)
	WRITE(6,999)
	WRITE(6,102)
	102				FORMAT('******** OUTFLOW START *******')
	WRITE(6,411)
	411				FORMAT('Time Step',5X,'CELL43',5X,'CELL98',5X,'CELL70',5X,'CELL79',5X,'CELL88')
	WRITE(6,560)
	560				FORMAT('--------------------------------------')
	DO 602 	J=1,EndSim,DT
		des2(J)=Y(70,J)+Y(79,J)+Y(88,J)
		des1(J)=Y(43,J)+Y(98,J)
		WRITE(6,515) J,Y(43,J),Y(98,J),Y(70,J),Y(79,J),Y(88,J)
		515							FORMAT (I5,5X,F6.2,5X,F6.2,5X,F6.2,5X,F6.2,5X,F6.2)
	602				CONTINUE
"""


import warnings

list_of_cells_where_output_printed = []
cell_numbers_where_outputs_printed = []
counter = 0


selected_cells_for_output = []
with open("input_output_text_files/selected_cells_for_output_plots.txt") as f:
    for row in f:
        cell_number = row.strip()
        selected_cells_for_output.append(int(cell_number))
selected_cells_for_output = set(selected_cells_for_output)  # for faster searching


# @Nishant: @pdoc had to switch to relative paths while generating the pdocs;
#           but we need the absolute path when running the main;  I think function
#           based script will solve the issues; When not using pdoc reset to absolute path
with open("input_output_text_files/input.txt") as f:
    for row in f:
        listed = row.strip().split("	")
        if len(listed) > 10:  # @Nishant: quickfix to filter out the input rows
            counter += 1

            if (which_type_of_cells in [listed[4], "ALL"]) or (counter in selected_cells_for_output):
                list_of_cells_where_output_printed.append(listed[0])
                cell_numbers_where_outputs_printed.append(counter)


#  @Nishant: Debug Output
# print("List of destinations: ", list_of_cells_where_output_printed)
# print("Serial number of cells: ", cell_numbers_where_outputs_printed)

print("		WRITE(6,999)")
print("		WRITE(6,999)")
print("		WRITE(6,102)")
print("		102				FORMAT('******** OUTFLOW START *******')")
print("! Comment from Nishant: Do not change  the above comment 'OUTFLOW START' because we use it to parse the output")
print("			WRITE(6,411)")

str_ = "	411				FORMAT('Time Step',5X,"

# To:do @Nishant: empty cell_numbers_where_outputs_printed → implies something wrong
# with the input file. no destination file need to put a check here!

for cell_num in cell_numbers_where_outputs_printed[:-1]:
    str_ += "'CELL" + str(cell_num) + "',5X,"

try:
    str_ += "'CELL" + str(cell_numbers_where_outputs_printed[-1]) + "')"
except:
    warnings.warn("Unable to generate fortran code due to no destination cells")
    sys.exit(-1)


print(str_)

print("     WRITE(6,560)")
print("    560				FORMAT('--------------------------------------')")
print("	DO 602 	J=1,EndSim,DT")

str_ = "	    WRITE(6,515) J,"
for cell_num in cell_numbers_where_outputs_printed[:-1]:
    str_ += "Y(" + str(cell_num) + ",J),"
str_ += "Y(" + str(cell_numbers_where_outputs_printed[-1]) + ",J)"
print(str_)


str_ = "		515							FORMAT (I5,"
for cell_num in cell_numbers_where_outputs_printed[:-1]:
    str_ += "5X,F6.4,"
str_ += "5X, F6.4)"
print(str_)


print("	602				CONTINUE")
print("     WRITE(6,562)")
print("     562				FORMAT('******** OUTFLOW END*******')")
print("! Comment from Nishant: Do not change  the above comment 'OUTFLOW END' because we use it to parse the output")
print("    STOP")
print("END")

# sys.exit(0)  # @Nishant:@pdoc we are using this so that we can return a value to the
# calling script, better to remove this and create a function like other files.
# @Nishant: had to comment out while generating the pdoc
