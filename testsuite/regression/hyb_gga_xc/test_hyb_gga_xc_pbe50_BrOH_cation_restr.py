
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbe50_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe50", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.052389110890199e+01, -1.052390566520676e+01, -1.052400289844775e+01, -1.052378566210550e+01, -1.052389486378334e+01, -1.052389486378334e+01, -1.778167071000314e+00, -1.778153224769101e+00, -1.777876023124527e+00, -1.778688364170826e+00, -1.778192358128529e+00, -1.778192358128529e+00, -3.761197418852266e-01, -3.758946016413782e-01, -3.714525457156895e-01, -3.740873601487929e-01, -3.733069394078815e-01, -3.733069394078815e-01, -1.139227705037987e-01, -1.150249452350033e-01, -4.324806101765715e-01, -9.313643455024576e-02, -1.009998766801854e-01, -1.009998766801854e-01, -5.043731065356849e-03, -5.308285302427817e-03, -2.884963996792372e-02, -2.914129441428206e-03, -3.661270433647007e-03, -3.661270433647007e-03, -2.576116234947637e+00, -2.576018900938779e+00, -2.576116115550768e+00, -2.576030101897890e+00, -2.576064302301477e+00, -2.576064302301477e+00, -1.077859102982157e+00, -1.083226614026957e+00, -1.077541723945907e+00, -1.082268497919426e+00, -1.080995221311783e+00, -1.080995221311783e+00, -3.291833852262227e-01, -3.573737514363582e-01, -3.069907897234180e-01, -3.200036154257833e-01, -3.339656451151813e-01, -3.339656451151813e-01, -6.998382272671336e-02, -1.188384572880773e-01, -6.532374124605295e-02, -9.786032694982274e-01, -7.869371817956625e-02, -7.869371817956625e-02, -2.248583585161243e-03, -2.848798930677500e-03, -2.178123976026170e-03, -4.552203614215764e-02, -2.623971779804055e-03, -2.623971779804055e-03, -3.358551076343445e-01, -3.319870620554781e-01, -3.332552479327345e-01, -3.343754565497127e-01, -3.338064184652266e-01, -3.338064184652266e-01, -3.286306298976877e-01, -2.823993346874726e-01, -2.931086343869257e-01, -3.053755987346424e-01, -2.988705681771511e-01, -2.988705681771511e-01, -3.724132810486619e-01, -1.436108348361159e-01, -1.652053743118851e-01, -2.062929944841932e-01, -1.837110982599142e-01, -1.837110982599142e-01, -2.619810584716553e-01, -2.766671497511965e-02, -3.728116895229678e-02, -1.994321466073350e-01, -5.624927918541651e-02, -5.624927918541653e-02, -7.114822587935949e-03, -7.616317816510045e-04, -1.598593147875441e-03, -5.325410076602378e-02, -2.428021533488447e-03, -2.428021533488448e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbe50_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe50", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.269854544575134e+01, -1.269859638189898e+01, -1.269880819464598e+01, -1.269805067444846e+01, -1.269844967793346e+01, -1.269844967793346e+01, -2.127571620553903e+00, -2.127600972907441e+00, -2.128329668775398e+00, -2.127448028602253e+00, -2.127655640806950e+00, -2.127655640806950e+00, -4.583558437357236e-01, -4.574368122648951e-01, -4.348004721614425e-01, -4.384789933699034e-01, -4.381579700109206e-01, -4.381579700109206e-01, -1.289256016319957e-01, -1.313960776170887e-01, -5.284287691245615e-01, -1.009500415133643e-01, -1.103866328806452e-01, -1.103866328806451e-01, -6.716756046161830e-03, -7.067869200947128e-03, -3.731051789779274e-02, -3.883992128902255e-03, -4.878312538121501e-03, -4.878312538121499e-03, -3.229376544713228e+00, -3.230942532206616e+00, -3.229446051598737e+00, -3.230828536008441e+00, -3.230171587720321e+00, -3.230171587720321e+00, -1.166474435195031e+00, -1.176127178791067e+00, -1.156984318258154e+00, -1.165404855900212e+00, -1.176537199928281e+00, -1.176537199928281e+00, -4.272477283359718e-01, -4.706744169584907e-01, -3.972684729574378e-01, -4.221620631332105e-01, -4.352181068113349e-01, -4.352181068113349e-01, -8.053211138210974e-02, -1.272185352230510e-01, -7.608646303872835e-02, -1.285087372433448e+00, -8.689100815756391e-02, -8.689100815756391e-02, -2.997312912263851e-03, -3.796864397715183e-03, -2.902448226417976e-03, -5.601826923519968e-02, -3.496580825335348e-03, -3.496580825335351e-03, -4.383835504021442e-01, -4.370151358645611e-01, -4.377574336876214e-01, -4.381696680350446e-01, -4.379878662058676e-01, -4.379878662058676e-01, -4.269956582695632e-01, -3.536418364066811e-01, -3.786454200375372e-01, -4.020341015695378e-01, -3.905275789840835e-01, -3.905275789840835e-01, -4.907092485305452e-01, -1.560039411426232e-01, -1.882126651229607e-01, -2.618086701538620e-01, -2.220745101366942e-01, -2.220745101366942e-01, -3.293353154523249e-01, -3.591292295478217e-02, -4.735134468590631e-02, -2.607260466961093e-01, -6.609908945348816e-02, -6.609908945348816e-02, -9.465751103609740e-03, -1.015444033738071e-03, -2.131039542469275e-03, -6.331759024827525e-02, -3.235538226492186e-03, -3.235538226492184e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbe50_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe50", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.332849479241366e-09, -3.332821405398269e-09, -3.332671520915342e-09, -3.333089709591412e-09, -3.332874153855724e-09, -3.332874153855724e-09, -3.678124074517048e-06, -3.678045267335479e-06, -3.675868405139863e-06, -3.677020746024723e-06, -3.677698427771333e-06, -3.677698427771333e-06, -1.239338848786537e-03, -1.255013014943767e-03, -1.608950774890292e-03, -1.558167791600149e-03, -1.562246240812726e-03, -1.562246240812726e-03, -1.564970355510769e-01, -1.469603349520096e-01, -7.243048960550532e-04, -3.069706653776980e-01, -2.522418141054376e-01, -2.522418141054378e-01, -1.214339999448810e+00, -1.282388318608002e+00, -6.917650544426686e-01, -1.105910775520223e+00, -1.403777582975234e+00, -1.403777582976634e+00, -7.547925389341424e-07, -7.530262472141035e-07, -7.547157157408899e-07, -7.531567270802516e-07, -7.538978564950140e-07, -7.538978564950140e-07, -3.055629689654278e-05, -2.986330129505307e-05, -3.083029938894709e-05, -3.022088161847665e-05, -3.002807265584288e-05, -3.002807265584288e-05, -1.276362918761530e-04, 2.545258916395428e-03, -3.524118752945239e-04, 3.557969148718454e-03, 1.472800965822635e-04, 1.472800965822635e-04, -4.265926911380135e-01, -1.372664241402275e-01, -4.762149382560077e-01, 6.189759204038630e-07, -4.599415750774875e-01, -4.599415750774875e-01, -1.457702989799103e+00, -1.301489937159304e+00, -8.166540515128453e+00, -7.405191053729828e-01, -3.800921015639090e+00, -3.800921015630160e+00, 5.657098704932426e-03, 3.913975639722385e-03, 4.465780933184726e-03, 4.970011492133074e-03, 4.712160063075111e-03, 4.712160063075111e-03, 7.100851576969907e-03, -2.575244542014930e-03, -5.332654286793297e-04, 1.896014376757038e-03, 6.215338393196056e-04, 6.215338393196056e-04, 1.875122736947289e-03, -7.103671867648617e-02, -3.820361709622559e-02, -5.545445402970250e-03, -1.882694921703985e-02, -1.882694921703984e-02, -3.120111084291818e-03, -6.337209815041398e-01, -6.182762057148748e-01, 3.400605159184521e-03, -7.716297418779612e-01, -7.716297418779645e-01, -9.767316732718521e-01, -6.394155668909601e+00, -3.093001864785163e+00, -7.933755620296096e-01, -4.801054930224270e+00, -4.801054930226116e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05