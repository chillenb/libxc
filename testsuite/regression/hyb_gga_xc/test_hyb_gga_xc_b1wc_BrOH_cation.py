
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b1wc_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.756060072852025e+01, -1.756062871548309e+01, -1.756080165057231e+01, -1.756038425903441e+01, -1.756059606805172e+01, -1.756059606805172e+01, -2.935354910322131e+00, -2.935336019961801e+00, -2.934970292619972e+00, -2.936155986164525e+00, -2.935402237030055e+00, -2.935402237030055e+00, -6.030838464073075e-01, -6.026992966602814e-01, -5.943607438319967e-01, -5.987350253036052e-01, -5.986733565866773e-01, -5.986733565866773e-01, -1.777356341111536e-01, -1.792407153250767e-01, -6.964657759135744e-01, -1.494363021074589e-01, -1.705480930623004e-01, -1.705480930623003e-01, -8.485218280042894e-03, -8.934372862509693e-03, -4.829757710530311e-02, -4.895705600030720e-03, -6.830868537211169e-03, -6.830868537211169e-03, -4.276193805459813e+00, -4.276053572066532e+00, -4.276194590335348e+00, -4.276070668185519e+00, -4.276118686392292e+00, -4.276118686392292e+00, -1.748043656282674e+00, -1.757758804688774e+00, -1.745346098216555e+00, -1.753917846320306e+00, -1.754744344317152e+00, -1.754744344317152e+00, -5.235029710130579e-01, -5.620174819315621e-01, -4.877651890937765e-01, -5.023759281739845e-01, -5.312197281732368e-01, -5.312197281732368e-01, -1.160047430401970e-01, -1.884742936557480e-01, -1.086135188384809e-01, -1.594380636572029e+00, -1.283585047979212e-01, -1.283585047979212e-01, -3.780028506137743e-03, -4.785791639119063e-03, -3.664416072791429e-03, -7.597719137833284e-02, -4.602847269823405e-03, -4.602847269823410e-03, -5.230811658673729e-01, -5.200544456167431e-01, -5.210549660929570e-01, -5.219293004367410e-01, -5.214856105155465e-01, -5.214856105155465e-01, -5.101792690833514e-01, -4.488512531021320e-01, -4.650244242482497e-01, -4.817797931433739e-01, -4.730704984951464e-01, -4.730704984951464e-01, -5.872786384482420e-01, -2.252137760693429e-01, -2.583818519320396e-01, -3.244031765279516e-01, -2.881214266544809e-01, -2.881214266544808e-01, -4.153638362376586e-01, -4.626567775072678e-02, -6.230893245617186e-02, -3.120913514624571e-01, -9.357962022433430e-02, -9.357962022433432e-02, -1.195140577086068e-02, -1.279511917317607e-03, -2.690507865913667e-03, -8.867943018403372e-02, -4.227142077657829e-03, -4.227142077657821e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b1wc_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.152436178787272e+01, -2.152433552551069e+01, -2.152444630342182e+01, -2.152439728323187e+01, -2.152471112311071e+01, -2.152477642518947e+01, -2.152375203234673e+01, -2.152355943161735e+01, -2.152440815101236e+01, -2.152404441880702e+01, -2.152440815101236e+01, -2.152404441880702e+01, -3.571974066394852e+00, -3.572004401049440e+00, -3.572000490908075e+00, -3.572029193758104e+00, -3.572621238200158e+00, -3.572764369911644e+00, -3.572108977406762e+00, -3.572239520495406e+00, -3.571392956873285e+00, -3.572813996119235e+00, -3.571392956873285e+00, -3.572813996119235e+00, -7.403668193421548e-01, -7.430351669067324e-01, -7.389629566189722e-01, -7.422304060742649e-01, -7.150967782834905e-01, -7.105506703743382e-01, -7.180426900713224e-01, -7.194838113232322e-01, -7.387594955339623e-01, -6.935734634625225e-01, -7.387594955339623e-01, -6.935734634625225e-01, -1.871995687954658e-01, -1.933951795631264e-01, -1.916653615984256e-01, -1.989211284424474e-01, -8.448873966387734e-01, -8.695978590734179e-01, -1.419270471695358e-01, -1.429626826843447e-01, -1.765959056051794e-01, -1.374541145536763e-01, -1.765959056051793e-01, -1.374541145536762e-01, -1.091956376855397e-02, -1.159930328053950e-02, -1.143832683964400e-02, -1.224777651548916e-02, -6.056776255332628e-02, -6.362347102836155e-02, -6.576447315664267e-03, -6.467171482754299e-03, -9.746934564917584e-03, -5.552826073358201e-03, -9.746934564917580e-03, -5.552826073358199e-03, -5.378780289609224e+00, -5.377521322738087e+00, -5.380787810600035e+00, -5.379467654463086e+00, -5.378890341024386e+00, -5.377590792366878e+00, -5.380617988102601e+00, -5.379353555041879e+00, -5.379806311014076e+00, -5.378499610771374e+00, -5.379806311014076e+00, -5.378499610771374e+00, -2.001393725256661e+00, -2.001297929874299e+00, -2.017158720039311e+00, -2.016634451896119e+00, -1.985656141410994e+00, -1.989347576299739e+00, -1.999385860197905e+00, -2.003206290455342e+00, -2.021140318805738e+00, -2.012006339795203e+00, -2.021140318805738e+00, -2.012006339795203e+00, -6.708496464751349e-01, -6.695923182571261e-01, -7.348806507778819e-01, -7.353860040192923e-01, -6.140446974852851e-01, -6.309693710020020e-01, -6.482243388135853e-01, -6.629199977492448e-01, -6.942444263943799e-01, -6.685058670567484e-01, -6.942444263943799e-01, -6.685058670567485e-01, -1.275296967018829e-01, -1.274291214807541e-01, -1.800827372446549e-01, -1.806466060585620e-01, -1.206240602185326e-01, -1.233147497101107e-01, -2.082359020448452e+00, -2.081589332452989e+00, -1.308444211166940e-01, -1.249342315691208e-01, -1.308444211166940e-01, -1.249342315691208e-01, -4.935276232187782e-03, -5.128930672285387e-03, -6.327585356573975e-03, -6.423419367437376e-03, -4.729958961744992e-03, -5.007934987002254e-03, -9.263451200580361e-02, -9.328320132781995e-02, -4.832704039492438e-03, -6.631014115313818e-03, -4.832704039492445e-03, -6.631014115313829e-03, -6.839606241432981e-01, -6.859322022673311e-01, -6.793238628102567e-01, -6.813491446641329e-01, -6.811012820766839e-01, -6.831287605540665e-01, -6.824787110719382e-01, -6.844537018490197e-01, -6.818045464433947e-01, -6.838054057909672e-01, -6.818045464433947e-01, -6.838054057909672e-01, -6.668071048507948e-01, -6.683683054362793e-01, -5.604928380419301e-01, -5.623265816411868e-01, -5.921218547547262e-01, -5.941377056211760e-01, -6.235335985540909e-01, -6.251584402925108e-01, -6.076998049825568e-01, -6.093650896396641e-01, -6.076998049825568e-01, -6.093650896396641e-01, -7.670650401113577e-01, -7.683898654429098e-01, -2.307771158711833e-01, -2.319439116931773e-01, -2.918503316150714e-01, -2.945384656241671e-01, -4.085436857804713e-01, -4.100882511653064e-01, -3.504741041016553e-01, -3.504459075135499e-01, -3.504741041016551e-01, -3.504459075135499e-01, -5.194702016326672e-01, -5.221341428902706e-01, -5.963143744856224e-02, -6.000360373588565e-02, -7.778749676143847e-02, -7.992387208585075e-02, -3.991928382801430e-01, -4.031632976118732e-01, -1.066883858662251e-01, -1.064977938970444e-01, -1.066883858662251e-01, -1.064977938970444e-01, -1.558470745448403e-02, -1.614304953333540e-02, -1.703908289692323e-03, -1.707718002638303e-03, -3.465916594890892e-03, -3.685990913737667e-03, -1.027040485700391e-01, -1.037783110327010e-01, -4.576250680145046e-03, -6.079454588372269e-03, -4.576250680145038e-03, -6.079454588372261e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b1wc_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.676267334053023e-09, 3.855567579455844e-10, -9.676325891827048e-09, -9.676233118666674e-09, 3.855637563905792e-10, -9.676300021559311e-09, -9.675975377900047e-09, 3.855734091993355e-10, -9.675976085429520e-09, -9.676378039570062e-09, 3.854697494807899e-10, -9.676473830584813e-09, -9.676271648386950e-09, 3.855271085313426e-10, -9.676153145097316e-09, -9.676271648386950e-09, 3.855271085313426e-10, -9.676153145097316e-09, -1.158523600061774e-05, 2.158879157656427e-06, -1.159172496821444e-05, -1.158558125011198e-05, 2.159323527660124e-06, -1.159253782723978e-05, -1.159905355381524e-05, 2.169371669283230e-06, -1.160361383237008e-05, -1.156856084417271e-05, 2.149943255801801e-06, -1.157421675585939e-05, -1.159128569682936e-05, 2.159156568071321e-06, -1.158487243897695e-05, -1.159128569682936e-05, 2.159156568071321e-06, -1.158487243897695e-05, -5.617118055876042e-03, 3.586683876242116e-03, -5.675697529694875e-03, -5.617382730325329e-03, 3.559577090514688e-03, -5.688143893445310e-03, -5.833676363343547e-03, 2.876045381924161e-03, -5.777659464018514e-03, -5.633892754145012e-03, 2.810968523930000e-03, -5.655844371335796e-03, -6.083589027232234e-03, 2.852692094259074e-03, -5.444260439610927e-03, -6.083589027232234e-03, 2.852692094259074e-03, -5.444260439610927e-03, -8.758710496581794e-01, 3.108902575321890e-01, -7.653137877989793e-01, -8.413311266335037e-01, 3.280826843588528e-01, -7.163093093751242e-01, -3.338878565805677e-03, 1.940414170439823e-03, -3.175153668034989e-03, -1.668918688781365e+00, 2.258949661124237e-01, -1.606935710961058e+00, -7.829838006852836e-01, 1.882246441728870e-01, -2.430254240326096e+00, -7.829838006852834e-01, 1.882246441728873e-01, -2.430254240326100e+00, -6.899392337112049e+00, 8.471332723898200e-03, -6.791705302073927e+00, -7.226859314844799e+00, 9.964515393423459e-03, -7.138306492736029e+00, -2.636938966421707e+00, 5.126941006152715e-02, -2.601757063607123e+00, -6.540232712529537e+00, 3.008683449168684e-03, -6.368738670758405e+00, -6.957751755796478e+00, 4.414566270491809e-03, -1.802088348742426e+01, -6.957751755796044e+00, 4.414566270088269e-03, -1.802088348742385e+01, -2.746204814274099e-06, 5.666948009101149e-07, -2.748884330772952e-06, -2.747605628468282e-06, 5.704413223836728e-07, -2.750179115552353e-06, -2.746275375886470e-06, 5.668443577206054e-07, -2.748892240521936e-06, -2.747431626975824e-06, 5.701514261968389e-07, -2.750117078813169e-06, -2.746957658927513e-06, 5.686004611431787e-07, -2.749527740671373e-06, -2.746957658927513e-06, 5.686004611431787e-07, -2.749527740671373e-06, -8.081887912325156e-05, 1.221811741084532e-05, -8.083404217722137e-05, -7.919260193092449e-05, 1.218856108878283e-05, -7.926454423003063e-05, -8.095205436631902e-05, 1.161795009824474e-05, -8.102723877946133e-05, -7.952523235448188e-05, 1.159619612758267e-05, -7.956931417148081e-05, -7.990023187193862e-05, 1.250231835492394e-05, -7.988328062862199e-05, -7.990023187193862e-05, 1.250231835492394e-05, -7.988328062862199e-05, -1.044766787650662e-02, 1.180452397706796e-02, -1.063711519144436e-02, -7.945673322890890e-03, 1.489938286408802e-02, -7.967963870630616e-03, -1.505499926780491e-02, 1.535577972946644e-02, -1.263175834871231e-02, -1.441027705968860e-02, 2.278859644535717e-02, -1.115365717420345e-02, -8.679935745441719e-03, 1.179162458306085e-02, -1.174484809809634e-02, -8.679935745441719e-03, 1.179162458306085e-02, -1.174484809809634e-02, -1.864579401963892e+00, 1.261488607621168e-01, -1.908933741763157e+00, -6.968009087938477e-01, 1.139695105238702e-01, -6.871639776540572e-01, -2.036421599065240e+00, 1.279112469154852e-01, -2.019861459133587e+00, -1.278771860707702e-04, 1.207887057881010e-04, -1.282478420711821e-04, -2.114902612547606e+00, 2.409472748596690e-01, -2.554232597670932e+00, -2.114902612547606e+00, 2.409472748596690e-01, -2.554232597670932e+00, -9.247489925439650e+00, 2.933546798259037e-03, -8.003260513481864e+00, -7.902894043811388e+00, 3.699144102256803e-03, -7.297913593046371e+00, -4.470431420037687e+01, 3.685108437354808e-02, -4.949932416432839e+01, -2.761642646550080e+00, 1.250822448857004e-01, -2.657252959386216e+00, -2.239222856814468e+01, 1.410360645889085e-02, -2.163233105471464e+01, -2.239222856814133e+01, 1.410360646869916e-02, -2.163233105471133e+01, -1.027550802859405e-02, 2.514346072918196e-02, -9.952351727606838e-03, -1.101322086158691e-02, 2.157065992117518e-02, -1.072106324286338e-02, -1.080200048287689e-02, 2.270457728491302e-02, -1.050178275308126e-02, -1.059310388067456e-02, 2.373865089872573e-02, -1.027926668544485e-02, -1.070234694833917e-02, 2.321019190891285e-02, -1.039521044605787e-02, -1.070234694833917e-02, 2.321019190891285e-02, -1.039521044605787e-02, -1.114526960206454e-02, 2.977269840520045e-02, -1.080826506488529e-02, -1.814516772226536e-02, 1.592277215307748e-02, -1.784850565942805e-02, -1.659083377046956e-02, 1.838749842245184e-02, -1.630891005109226e-02, -1.508530685772039e-02, 2.169730037966074e-02, -1.477955384026332e-02, -1.591548972365891e-02, 1.995497716545398e-02, -1.559826842641651e-02, -1.591548972365891e-02, 1.995497716545398e-02, -1.559826842641651e-02, -6.709819253091594e-03, 1.183264719377924e-02, -6.690880114589998e-03, -3.288383044711926e-01, 8.498631964250428e-02, -3.224283427637853e-01, -1.692343339849947e-01, 7.821972418059023e-02, -1.632451601799315e-01, -6.348046474356382e-02, 6.964082434152352e-02, -6.182319966732042e-02, -9.873801533553536e-02, 7.796350422599263e-02, -9.939942769555002e-02, -9.873801533553547e-02, 7.796350422599266e-02, -9.939942769555002e-02, -2.457081695711963e-02, 2.285950305464996e-02, -2.398874747691023e-02, -2.450221846363765e+00, 4.213028450134424e-02, -2.446824072641908e+00, -2.200166330393564e+00, 6.458066222247318e-02, -2.232558982144375e+00, -7.845617245498623e-02, 1.099497136873529e-01, -7.291177634505075e-02, -2.895060453347952e+00, 2.182010199987041e-01, -3.560418052414847e+00, -2.895060453347952e+00, 2.182010199987039e-01, -3.560418052414848e+00, -5.278662680866920e+00, 1.056259116938041e-02, -5.356924970944378e+00, -2.882777019429012e+01, 4.011531506685853e-03, -5.099847776191064e+01, -1.764744933968874e+01, 5.020763908060611e-03, -1.875362801060292e+01, -3.230044287143490e+00, 2.002688123716920e-01, -3.175474640179830e+00, -4.590607000372627e+01, 1.800188742308661e-02, -2.256609540183874e+01, -4.590607000372267e+01, 1.800188743452388e-02, -2.256609540183514e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05