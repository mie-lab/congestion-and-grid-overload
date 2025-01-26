PROGRAM CTMSIM
    !	USE MSFLIB
    !	USE PORTLIB

    !----------------
    ! DECLARE VARIABLES
    !----------------
    IMPLICIT NONE
    REAL V(200)						! Free flow speed (kph) Array(flow)
    REAL W(200)						! Shockwave speed (kph)	Array(flow)
    REAL Kj					    ! Jam Density (veh/km-lane)
    REAL SatFlow(200)				! per lane saturation flow (vph)
    REAL TotVol1(50),TotVol2(50),Vol1,Vol2 ! Cumulative volume at each 15 minute
    INTEGER DT					! Length of time step (sec)
    INTEGER EndSim				! Actual time step of simulation
    INTEGER NumCell 			! Number of cells in the network
    CHARACTER Name(200)*11		! Name of the cells Array(i)
    REAL FULLfactor(200)		! Proportion of cell that is full at t=0
    REAL Lanes(200)			    ! Number of lanes _leaving_ cell(i)
    !		Origin cells should have a number of lanes
    !		Destination should have zero lanes.
    REAL TURNfactor(200)		! Turning Factor, adjust SatFlow
    !		Qmax(I)= SatFlow*Lanes(I)*TURNfactor(I)
    !		If cell is start of diverge, the program calculates
    !		that cell's Qmax by looking at the Qmaxs of the downstream
    !		branches, overwriting whatever initial value the cell has
    CHARACTER CellType(200)*2	! Type of Cell
    !		OR=	Origin				Create holding capacity of 99,999
    !		TR=	Transition			Not examined
    !		DE=	Destination			Create holding cap of 99,999, set delay array=0
    !		DI=	Start of Diverge	Split cell's outflow
    !		DB= Diverge Branch	    Not examined
    !		ME=	(UnSig)Merge Branch Inflow of downstream cell is sum of MB cells
    !		MB=	(Sig) Merge Branch	Inflow of downstream cell is sum of MB cells
    CHARACTER NameFrom1(200)*11	! Name of First 'from' link
    CHARACTER NameFrom2(200)*11	! Name of Second 'from' link
    CHARACTER NameTo1(200)*11	! Name of First 'to' link
    CHARACTER NameTo2(200)*11	! Name of Second 'to' link
    INTEGER From1(200)			! First 'from' link
    INTEGER From2(200)			! Second 'from' link
    INTEGER To1(200)			! First 'to' link
    INTEGER To2(200)			! Second 'to' link
    REAL Split1(200)			! Proportion to first branch
    ! Currently: For head of diverge only
    REAL Split2(200)			! Proportion to second branch
    REAL Priority(200)			! Priority of the unsignalize merge
    CHARACTER SigQuery(200)		! Signal regulating outflow? ('Y' or 'y'=yes)
    INTEGER Offset(200)			! Signal offset
    INTEGER Greff(200)			! Effecetive green duration
    INTEGER Redeff(200)			! Effecetive green duration
    INTEGER CYC					! Cycle length
    INTEGER IntoCycle           ! Time into cycle -phase at T
    INTEGER T					! time step index
    INTEGER I					! cell number index
    REAL Q						! Max cell flow
    INTEGER NumLoad				! Number of loadings
    CHARACTER NameLoadCell*11	! Name of Cell being loaded
    INTEGER LoadCell(15)			! Cell being loaded
    REAL Dem1(8000),Dem2(8000),Dem3(8000) ! Demand into cell (veh/hour)
    REAL Dem4(8000),Dem5(8000),Dem6(8000)
    INTEGER LoadSt				! Start time step of loading
    INTEGER LoadEnd				! End time step of loading
    INTEGER b,J,K
    INTEGER NN					!Number of 15 minute intervals
    INTEGER Count				! Generic counting variable
    REAL Send
    REAL Recieve
    INTEGER OppNo

    REAL MaxFactor(200)
    REAL HoldCap (200)			! Cell holding capacity array
    !		NOTE:  If the cell is an origin or destination, the program
    !		assigns a holing capacity of 99,999 vehicles.
    REAL Qmax(200)				! Max outflow from cell(i)
    REAL N(200,0:9999)			! Cell occupancy array (i,t)
    REAL S(200,0:9999)			! Cell outflows (i,t)
    REAL Y(200,0:9999),des1(0:9999),des2(0:9999)			! Cell inflows array (i,t)
    COMMON/BLOCK1/V,W,Split1,Split2,HoldCap,Qmax,N,S,Y,&
            Priority,DT,EndSim,NumCell,From1,From2,To1,To2


    ! ------------------------------
    !	TITLE BAR
    ! ------------------------------
    PRINT*,'   ** CTM SIMULATION FOR DUBLIN STREETS **'

    ! ------------------------------
    !	Read and Write Global Parameters
    ! ------------------------------
    PRINT*,'Opening Scenario File'
    OPEN (2,FILE='input.txt')	! Scenario File
    OPEN (3,FILE='out30.txt')
    OPEN (6,FILE='outfl.txt')	! Output File
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
        WRITE (6,203) I,Qmax(I),HoldCap(I),N(I,0)
        203		FORMAT ('Cell:',I5,3X,'Qmax:',F8.2,3X,'Capacity:',F7.2,&
                3X,'Initial n:',F8.2)
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


    DO 70 T = LoadSt,LoadEnd
        READ(3,*)Dem1(T),Dem2(T),Dem3(T),Dem4(T),Dem5(T),Dem6(T)
        Y(LoadCell(1),T)=Dem1(T)*DT
        Y(LoadCell(2),T)=Dem2(T)*DT
        Y(LoadCell(3),T)=Dem3(T)*DT
        Y(LoadCell(4),T)=Dem4(T)*DT
        Y(LoadCell(5),T)=Dem5(T)*DT
        Y(LoadCell(6),T)=Dem6(T)*DT

    70	CONTINUE


    ! ****************************************
    ! Start of CTM Simulation
    ! ****************************************
    ! -----------------------
    ! Determine cell outflows
    ! -----------------------
    DO 200 T=0,EndSim,DT

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
            !210			CONTINUE
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

    !	--------------
    !	Outflow from main street

    WRITE(6,999)
    WRITE(6,999)
    WRITE(6,102)
    102				FORMAT('******** O U T F L O W  *******')
    WRITE(6,411)
    411				FORMAT('Time Step',5X,'CELL43',5X,'CELL98',5X,'CELL70',5X,'CELL79',5X,'CELL88')
    WRITE(6,560)
    560				FORMAT('--------------------------------------')
    DO 602 	J=1,EndSim,DT

        !@Nishant commented: I don't think this is being used

        !							des2(J)=Y(70,J)+Y(79,J)+Y(88,J)
        !							des1(J)=Y(43,J)+Y(98,J)
        WRITE(6,515) J,Y(43,J),Y(98,J),Y(70,J),Y(79,J),Y(88,J)
        515							FORMAT (I5,5X,F6.2,5X,F6.2,5X,F6.2,5X,F6.2,5X,F6.2)
    602				CONTINUE

    WRITE(6,562)
    562				FORMAT('******** O U T F L O W  END *******')

    ! @Nishant Commented out the cumulative flows
    !	number of steps in each 2 minute interval



    !				NN = (2*60)/DT
    !				WRITE(6,152)
    !152				FORMAT('******** C U M U L A T I V E  O U T F L O W  *******')
    !				WRITE(6,488)
    !488				FORMAT('Time Step',5X,'Volume1',5X,'Volume2')
    !				WRITE(6,588)
    !588				FORMAT('--------------------------------------')
    !
    !				DO K=1,(EndSim/NN)
    !					TotVol1(K)=0
    !					TotVol2(K)=0
    !					   DO J=(1+NN*(K-1)),(NN*K),DT
    !						 TotVol1(K)=TotVol1(K)+des1(J)
    !						 TotVol2(K)=TotVol2(K)+des2(J)
    !					   ENDDO
    !					 WRITE(6,527) K,TotVol1(K),TotVol2(K)
    !527					 FORMAT (I5,5X,F10.4,5X,F10.4)
    !				ENDDO
    !
    !				Vol1=TotVol1(1)
    !				Vol2=TotVol2(1)
    !				DO K=2,30
    !					Vol1=Vol1+TotVol1(K)
    !					Vol2=Vol2+TotVol2(K)
    !				ENDDO
    !				PRINT*,Vol1,Vol2


    !	----------------
    !	Close output file, end program
    CLOSE (6)

    STOP
END








