
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pw_rpa_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_rpa", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.883717752327060e-01, -1.883718609513650e-01, -1.883722684899252e-01, -1.883709926001670e-01, -1.883716573710046e-01, -1.883716573710046e-01, -1.333765545999758e-01, -1.333766185320676e-01, -1.333788220889847e-01, -1.333806156436566e-01, -1.333773344621899e-01, -1.333773344621899e-01, -8.674784006981150e-02, -8.671813929587248e-02, -8.600268109076059e-02, -8.622388419909237e-02, -8.597289393782573e-02, -8.597289393782573e-02, -5.383147538612706e-02, -5.414432313420583e-02, -9.080771383410230e-02, -4.809656816174879e-02, -4.386207955881088e-02, -4.386207955881088e-02, -9.397715770078620e-03, -9.695334733014359e-03, -2.570891926110628e-02, -6.669389874980288e-03, -7.309231668215170e-03, -7.309231668215170e-03, -1.454131347603905e-01, -1.454186426641739e-01, -1.454134089360987e-01, -1.454182714729847e-01, -1.454159155983098e-01, -1.454159155983098e-01, -1.165510550896363e-01, -1.167597289541856e-01, -1.163976046916509e-01, -1.165830016462356e-01, -1.167388286070480e-01, -1.167388286070480e-01, -8.321695975093413e-02, -8.539667007443945e-02, -8.113447678341802e-02, -8.215248291746363e-02, -8.351558179589746e-02, -8.351558179589744e-02, -4.102473620336606e-02, -5.390350907187658e-02, -3.955410007722153e-02, -1.159939635935764e-01, -4.397675341171206e-02, -4.397675341171206e-02, -5.646406474680125e-03, -6.573867930906475e-03, -5.530035544343272e-03, -3.284958791886537e-02, -6.120652091386021e-03, -6.120652091386021e-03, -8.334349707023812e-02, -8.319277506908326e-02, -8.324589516796994e-02, -8.328958900447121e-02, -8.326772652956051e-02, -8.326772652956051e-02, -8.262463917960507e-02, -7.867201435996971e-02, -7.985741919864896e-02, -8.098807620897275e-02, -8.041615588833313e-02, -8.041615588833313e-02, -8.664826617080516e-02, -5.907379316195763e-02, -6.328422656674620e-02, -6.994533195242476e-02, -6.656691611584394e-02, -6.656691611584394e-02, -7.656085182848227e-02, -2.515605663779349e-02, -2.948235520703492e-02, -6.904434089200781e-02, -3.681681369797939e-02, -3.681681369797939e-02, -1.159443511379744e-02, -2.758520839073535e-03, -4.519735535293879e-03, -3.578702743928027e-02, -5.845001428052518e-03, -5.845001428052522e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pw_rpa_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_rpa", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.986169626889833e-01, -1.986173459809657e-01, -1.986169171592476e-01, -1.986175634897758e-01, -1.986180064323114e-01, -1.986172918636672e-01, -1.986152355983352e-01, -1.986175028967384e-01, -1.986151392991961e-01, -1.986189329741462e-01, -1.986151392991961e-01, -1.986189329741462e-01, -1.432492017998018e-01, -1.432520282391991e-01, -1.432490659908751e-01, -1.432522934900741e-01, -1.432550426064156e-01, -1.432507778366409e-01, -1.432561793100249e-01, -1.432532719569614e-01, -1.432901735444825e-01, -1.432126772363974e-01, -1.432901735444825e-01, -1.432126772363974e-01, -9.586354508495304e-02, -9.535267703344824e-02, -9.589171559844409e-02, -9.526450629019644e-02, -9.439765423934737e-02, -9.528164696469167e-02, -9.520349231704327e-02, -9.492595568960752e-02, -9.100911094622965e-02, -9.938621110877457e-02, -9.100911094622965e-02, -9.938621110877457e-02, -6.249751224019921e-02, -5.970147513488104e-02, -6.305935188163729e-02, -5.985532460939116e-02, -1.021583738842799e-01, -9.766027062202712e-02, -5.530378194797544e-02, -5.443270967034695e-02, -4.412136332543560e-02, -8.523692228732842e-02, -4.412136332543561e-02, -8.523692228732839e-02, -1.153527313769373e-02, -1.114919379604131e-02, -1.192860916648681e-02, -1.147085390827328e-02, -3.097891504745405e-02, -2.948840977604720e-02, -8.052796622566600e-03, -8.118039545740622e-03, -8.421955305628446e-03, -1.086335389099842e-02, -8.421955305628446e-03, -1.086335389099842e-02, -1.553878454575983e-01, -1.554422419219320e-01, -1.553926857204512e-01, -1.554485194797689e-01, -1.553876162766870e-01, -1.554430251932868e-01, -1.553929389025067e-01, -1.554475162218513e-01, -1.553901531026320e-01, -1.554455476482029e-01, -1.553901531026320e-01, -1.554455476482029e-01, -1.261691836769737e-01, -1.261782383836193e-01, -1.263617140153228e-01, -1.264105790556665e-01, -1.261724797916972e-01, -1.258632721321203e-01, -1.263679853845378e-01, -1.260453204899786e-01, -1.259734522462081e-01, -1.267612719780954e-01, -1.259734522462081e-01, -1.267612719780954e-01, -9.178496952996462e-02, -9.211147146425253e-02, -9.423639241788111e-02, -9.417881753252166e-02, -9.205732162825842e-02, -8.773638560496066e-02, -9.291973389128932e-02, -8.895167819801318e-02, -8.932610197541839e-02, -9.565311287188788e-02, -8.932610197541838e-02, -9.565311287188784e-02, -4.742995306299877e-02, -4.693892220826942e-02, -6.128438157863038e-02, -6.091918753264082e-02, -4.695925658253730e-02, -4.437670444954427e-02, -1.255664884295870e-01, -1.256465910659190e-01, -5.216565117628873e-02, -4.889737044783562e-02, -5.216565117628873e-02, -4.889737044783562e-02, -6.923702040501811e-03, -6.805760416695256e-03, -7.999801712640708e-03, -7.942664305108363e-03, -6.814318129761721e-03, -6.643693266794592e-03, -3.829537876842457e-02, -3.805988420448491e-02, -8.196117515475190e-03, -7.115210933182300e-03, -8.196117515475190e-03, -7.115210933182300e-03, -9.231822572591282e-02, -9.184221456964616e-02, -9.216374789726019e-02, -9.168394399713985e-02, -9.221896150171750e-02, -9.173897581405992e-02, -9.226222929276222e-02, -9.178633940984472e-02, -9.224055533872791e-02, -9.176266518543896e-02, -9.224055533872791e-02, -9.176266518543896e-02, -9.153533665266501e-02, -9.113218425321529e-02, -8.747377743335331e-02, -8.697466331038632e-02, -8.871845782374013e-02, -8.819829290391897e-02, -8.984322513721096e-02, -8.942416853835849e-02, -8.925866218678906e-02, -8.881960015046908e-02, -8.925866218678906e-02, -8.881960015046908e-02, -9.560883941841669e-02, -9.539885344555028e-02, -6.685360029278446e-02, -6.637940003859034e-02, -7.153044475986942e-02, -7.063930110842864e-02, -7.837707012498084e-02, -7.782220032180209e-02, -7.452709729203363e-02, -7.456057380610777e-02, -7.452709729203365e-02, -7.456057380610777e-02, -8.540941472412245e-02, -8.464167820419406e-02, -2.964226085868817e-02, -2.946154463693401e-02, -3.499175305300690e-02, -3.390141602271959e-02, -7.790847642626608e-02, -7.642523490634237e-02, -4.387894491128196e-02, -4.143922645196367e-02, -4.387894491128197e-02, -4.143922645196367e-02, -1.407647252292587e-02, -1.376933320218564e-02, -3.381833095329934e-03, -3.379268088040000e-03, -5.583764355583652e-03, -5.445830556815882e-03, -4.178754408522612e-02, -4.108762857769484e-02, -7.726737018443143e-03, -6.818074722702850e-03, -7.726737018443146e-03, -6.818074722702852e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05