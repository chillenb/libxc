
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_n12_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.350688779368487e-01, -1.350690443171633e-01, -1.350696498310865e-01, -1.350671796697989e-01, -1.350684923108610e-01, -1.350684923108610e-01, -9.157065627682201e-02, -9.156991603711759e-02, -9.155339633295255e-02, -9.158661296839192e-02, -9.157032932854033e-02, -9.157032932854033e-02, -5.562527573849138e-02, -5.554015582970356e-02, -5.206829591562717e-02, -5.246341634254648e-02, -4.864724566569477e-02, -4.864724566569477e-02, -1.561195987270884e-02, -1.591843852717935e-02, -5.889154447332244e-02, -3.292027243446146e-02, -3.793221978486296e-02, -3.793221978486314e-02, -2.673872794564644e-02, -2.790431207384087e-02, -9.704018474408420e-02, -1.649360224146996e-02, -1.320181925114642e-02, -1.320181925114643e-02, -1.031922514180438e-01, -1.033091709297998e-01, -1.031970610307390e-01, -1.033002389434064e-01, -1.032514325950014e-01, -1.032514325950014e-01, -7.313777790069334e-02, -7.383747295241018e-02, -7.168912974911634e-02, -7.235756458451832e-02, -7.415571294050355e-02, -7.415571294050355e-02, -5.263433993539351e-02, -6.415345072603146e-02, -5.126459751572228e-02, -5.821007752001590e-02, -5.275963890416393e-02, -5.275963890416377e-02, -8.491658472984409e-02, -2.348296353559119e-02, -9.008801856861659e-02, -9.240351213506866e-02, -5.306353774864252e-02, -5.306353774864252e-02, -1.298703724416512e-02, -1.615702209565818e-02, -1.256059328177610e-02, -1.036890575502358e-01, -1.288727806479553e-02, -1.288727806479555e-02, -5.915719305039506e-02, -6.256094076835524e-02, -6.356724290034087e-02, -6.295601491329268e-02, -6.348745852871920e-02, -6.348745852871920e-02, -5.426122133002428e-02, -4.968313974342210e-02, -5.050264287806509e-02, -5.162854406352876e-02, -5.051989268177207e-02, -5.051989268177207e-02, -6.443015278288527e-02, -1.758775881523265e-02, -2.307969900657995e-02, -4.245636789234917e-02, -3.384446087880817e-02, -3.384446087880824e-02, -4.800782209377645e-02, -9.608626734740268e-02, -1.070513255174465e-01, -4.277464412890209e-02, -8.753393840663877e-02, -8.753393840663860e-02, -3.605967701771100e-02, -4.704126403173326e-03, -9.427846261475858e-03, -9.192217582137761e-02, -1.234456259822649e-02, -1.234456259822653e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_n12_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.520707838701523e-01, -1.520717647729340e-01, -1.520750916016219e-01, -1.520747835832542e-01, -1.520830183075721e-01, -1.520879613809997e-01, -1.520359374906738e-01, -1.520263059188880e-01, -1.520770693289930e-01, -1.520437684597653e-01, -1.520770693289930e-01, -1.520437684597653e-01, -8.901611353472708e-02, -8.908795374844825e-02, -8.901589232581242e-02, -8.909275308798781e-02, -8.908092735648471e-02, -8.914049943921694e-02, -8.898664912810118e-02, -8.905433203240623e-02, -8.904699451334527e-02, -8.906402416994770e-02, -8.904699451334527e-02, -8.906402416994770e-02, -7.675137897152673e-02, -7.211947640196686e-02, -7.821638624861225e-02, -7.240172305557797e-02, -9.252506264949899e-02, -1.041710992927824e-01, -9.927446941791676e-02, -9.550419270765875e-02, -5.233010065848252e-02, -1.682503197420688e-01, -5.233010065848252e-02, -1.682503197420688e-01, -2.261593468210218e-02, -1.871376337522911e-02, -3.468184008513898e-02, -2.736081396585111e-02, -8.208668874455051e-02, -6.215775897743325e-02, 1.096403876432607e-01, 1.018462382345277e-01, 1.194747291015535e-03, 5.278228937977137e-01, 1.194747291014992e-03, 5.278228937977111e-01, -3.787621135624453e-02, -3.118217203869375e-02, -3.999380156947895e-02, -3.208988984776632e-02, -9.142654907816612e-02, -7.957883112266947e-02, -2.086460550999987e-02, -2.202167369139989e-02, -1.035502715662218e-02, -5.362547379329355e-02, -1.035502715662175e-02, -5.362547379329410e-02, -1.482569423278331e-01, -1.482720742141047e-01, -1.491412408994803e-01, -1.491203009526768e-01, -1.483039414043850e-01, -1.482966626855635e-01, -1.490581865597198e-01, -1.490719976341114e-01, -1.487141102489204e-01, -1.486971403845657e-01, -1.487141102489204e-01, -1.486971403845657e-01, -1.442604683799184e-01, -1.442974978193136e-01, -1.418460558995026e-01, -1.420803210511850e-01, -1.518397743574315e-01, -1.476108656063876e-01, -1.499138792441782e-01, -1.457571282874849e-01, -1.357460247590872e-01, -1.447195469311311e-01, -1.357460247590872e-01, -1.447195469311311e-01, -5.600083783797076e-02, -5.664709650681327e-02, -9.339405695050920e-02, -9.223348209379190e-02, -5.518694031326065e-02, -5.056804883662846e-02, -9.759336353377865e-02, -9.345738505385344e-02, -5.746098733723601e-02, -6.376107664038988e-02, -5.746098733723392e-02, -6.376107664038834e-02, 9.577304662616520e-02, 8.672237708942677e-02, 7.075821669717892e-02, 6.858734400975654e-02, 8.966963701295566e-02, 5.839019056146909e-02, -1.268402594646159e-01, -1.268118951850235e-01, 1.630380445352876e-01, 8.875390630660640e-02, 1.630380445352876e-01, 8.875390630660640e-02, -1.805492362098733e-02, -1.594808842240172e-02, -2.151444605214567e-02, -2.049747522974620e-02, -1.798783422821096e-02, -1.496430991961995e-02, -1.728188291045522e-02, -1.610363882696743e-02, -3.062495243012673e-02, -1.140452262316603e-02, -3.062495243012643e-02, -1.140452262316581e-02, -4.323092955869082e-02, -4.264817161071000e-02, -8.855042812540041e-02, -8.725883429040726e-02, -7.426081583685125e-02, -7.243463589283065e-02, -5.782754240696257e-02, -5.627539368450208e-02, -6.629583570288858e-02, -6.453883969382374e-02, -6.629583570288858e-02, -6.453883969382374e-02, -5.655413985388688e-02, -5.637417543389739e-02, -6.596859507775210e-02, -6.409056324346166e-02, -5.183903352559123e-02, -5.124100419273105e-02, -7.118696445105549e-02, -7.130660281879662e-02, -5.597529972161070e-02, -5.574566319116556e-02, -5.597529972161070e-02, -5.574566319116556e-02, -9.856358926782585e-02, -9.732059160018669e-02, -8.893250115283378e-03, -8.827339613812116e-03, -7.739565163052836e-02, -7.202744028470702e-02, -6.730938372811300e-02, -6.510259272935683e-02, -8.877328052668765e-02, -8.793019334288640e-02, -8.877328052668972e-02, -8.793019334289193e-02, -6.636875993928137e-02, -6.313316330992423e-02, -8.946141426882083e-02, -8.784610675705594e-02, -6.876282093389348e-02, -6.605590177311249e-02, -5.065907168107924e-02, -4.669085136209242e-02, 8.037689056856943e-02, 3.577536054288392e-02, 8.037689056856763e-02, 3.577536054288378e-02, -4.821483201294267e-02, -4.303058573391925e-02, -6.223395421862924e-03, -6.178104604222117e-03, -1.369584918472609e-02, -1.122551729309954e-02, 4.283630867496625e-02, 3.714802086330420e-02, -2.740356622245908e-02, -1.122434915813410e-02, -2.740356622245944e-02, -1.122434915813386e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_n12_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.416729051514419e-11, 0.000000000000000e+00, 4.417052686509048e-11, 4.418987746321512e-11, 0.000000000000000e+00, 4.418468298422607e-11, 4.422298960538955e-11, 0.000000000000000e+00, 4.425371998921342e-11, 4.399808182486025e-11, 0.000000000000000e+00, 4.393344561840489e-11, 4.421363569485997e-11, 0.000000000000000e+00, 4.401218774056074e-11, 4.421363569485997e-11, 0.000000000000000e+00, 4.401218774056074e-11, -3.321938513230934e-07, 0.000000000000000e+00, -3.296715923198651e-07, -3.322411907794057e-07, 0.000000000000000e+00, -3.295532124655806e-07, -3.311561029560322e-07, 0.000000000000000e+00, -3.287281016045135e-07, -3.325106194402929e-07, 0.000000000000000e+00, -3.298484195796056e-07, -3.331511243682218e-07, 0.000000000000000e+00, -3.285413955567174e-07, -3.331511243682218e-07, 0.000000000000000e+00, -3.285413955567174e-07, 9.574509066696184e-04, 0.000000000000000e+00, 7.673626929211034e-04, 1.043526946637060e-03, 0.000000000000000e+00, 8.043316166838463e-04, 2.160521863955148e-03, 0.000000000000000e+00, 2.619312981347685e-03, 2.320773257714692e-03, 0.000000000000000e+00, 2.178364791316958e-03, 5.435437653496863e-04, 0.000000000000000e+00, 4.747873770078682e-03, 5.435437653496863e-04, 0.000000000000000e+00, 4.747873770078682e-03, 8.602875919187931e-02, 0.000000000000000e+00, -5.651569069521246e-03, 2.688036082841228e-01, 0.000000000000000e+00, 1.102861635571014e-01, 4.253040243078235e-04, 0.000000000000000e+00, 2.072187291782331e-05, -4.127725721310857e+00, 0.000000000000000e+00, -3.716104740418093e+00, -5.088644184367975e-01, 0.000000000000000e+00, -4.263219633260073e+01, -5.088644184368221e-01, 0.000000000000000e+00, -4.263219633260066e+01, -1.359423211472982e+02, 0.000000000000000e+00, -8.390256463089275e+01, -1.471711909524271e+02, 0.000000000000000e+00, -8.518027763602514e+01, -5.247004630205829e+01, 0.000000000000000e+00, -3.403028134748399e+01, -9.598176207907072e+01, 0.000000000000000e+00, -1.096629914982586e+02, -1.537380426837254e+01, 0.000000000000000e+00, -1.329338778103911e+03, -1.537380426800525e+01, 0.000000000000000e+00, -1.329338778102227e+03, 2.987850211440222e-07, 0.000000000000000e+00, 2.987001302080569e-07, 3.071221396842515e-07, 0.000000000000000e+00, 3.066950516947739e-07, 2.992251728969064e-07, 0.000000000000000e+00, 2.989258301224952e-07, 3.063300856746065e-07, 0.000000000000000e+00, 3.062372305255634e-07, 3.030857993109568e-07, 0.000000000000000e+00, 3.026967968603633e-07, 3.030857993109568e-07, 0.000000000000000e+00, 3.026967968603633e-07, 1.483991640516542e-05, 0.000000000000000e+00, 1.484988603932495e-05, 1.392519046910335e-05, 0.000000000000000e+00, 1.398060306244498e-05, 1.637355939911241e-05, 0.000000000000000e+00, 1.571811579892668e-05, 1.559872768581758e-05, 0.000000000000000e+00, 1.495425952136142e-05, 1.299393191381699e-05, 0.000000000000000e+00, 1.445855106484917e-05, 1.299393191381699e-05, 0.000000000000000e+00, 1.445855106484917e-05, -7.921583117122281e-04, 0.000000000000000e+00, -7.274575295544531e-04, 1.419083011619812e-02, 0.000000000000000e+00, 1.362791378547497e-02, -2.061135047417135e-03, 0.000000000000000e+00, -1.461121667546066e-03, 2.909835321198521e-02, 0.000000000000000e+00, 2.477993686958862e-02, 4.522291210025123e-04, 0.000000000000000e+00, 1.211330406087435e-04, 4.522291210031378e-04, 0.000000000000000e+00, 1.211330406087427e-04, -1.345165667354827e+01, 0.000000000000000e+00, -1.248644029313747e+01, -8.962377428695023e-01, 0.000000000000000e+00, -8.648899771921249e-01, -1.944316791866086e+01, 0.000000000000000e+00, -1.286006771783046e+01, 7.284855669503861e-05, 0.000000000000000e+00, 7.273505279251157e-05, -1.189400711285859e+01, 0.000000000000000e+00, -7.660306966855879e+00, -1.189400711285859e+01, 0.000000000000000e+00, -7.660306966855879e+00, -1.613977302140248e+02, 0.000000000000000e+00, -1.187622450385729e+02, -1.285672754624072e+02, 0.000000000000000e+00, -1.140410842194618e+02, -9.603061699445396e+02, 0.000000000000000e+00, -6.095197797902873e+02, -3.562299034416166e+01, 0.000000000000000e+00, -3.392074033346531e+01, -1.095292670194889e+03, 0.000000000000000e+00, -8.891849964590227e+01, -1.095292670196800e+03, 0.000000000000000e+00, -8.891849964702344e+01, -6.787158327218776e-02, 0.000000000000000e+00, -6.799913617487846e-02, 1.923729269893169e-02, 0.000000000000000e+00, 1.825301049514598e-02, 3.530664691464299e-03, 0.000000000000000e+00, 1.701348886377854e-03, -2.355177870038346e-02, 0.000000000000000e+00, -2.527023293235798e-02, -8.201895053734894e-03, 0.000000000000000e+00, -1.009323259289075e-02, -8.201895053734894e-03, 0.000000000000000e+00, -1.009323259289075e-02, -3.280605140553501e-02, 0.000000000000000e+00, -3.065893288897795e-02, 3.306686261153920e-03, 0.000000000000000e+00, 2.871376984438955e-03, -2.304053636477454e-03, 0.000000000000000e+00, -2.209370091912076e-03, 7.883344824989116e-03, 0.000000000000000e+00, 8.171817440668807e-03, -5.925993787814033e-04, 0.000000000000000e+00, -4.235081048153223e-04, -5.925993787814033e-04, 0.000000000000000e+00, -4.235081048153223e-04, 1.251026819825900e-02, 0.000000000000000e+00, 1.218677336861446e-02, -4.780603696398244e-02, 0.000000000000000e+00, -4.894018402860066e-02, 1.351854779587353e-01, 0.000000000000000e+00, 1.187946002529290e-01, 3.154505421155306e-02, 0.000000000000000e+00, 2.837772451528139e-02, 1.030417840995987e-01, 0.000000000000000e+00, 1.026040349362245e-01, 1.030417840996032e-01, 0.000000000000000e+00, 1.026040349362240e-01, 5.863506797301205e-03, 0.000000000000000e+00, 4.744145233002198e-03, -4.068403939203513e+01, 0.000000000000000e+00, -3.853840933892005e+01, -3.831677070211803e+01, 0.000000000000000e+00, -2.950596398147128e+01, 2.343198892435808e-03, 0.000000000000000e+00, -2.891519571814401e-03, -3.482592419504066e+01, 0.000000000000000e+00, -2.272130367780896e+01, -3.482592419504065e+01, 0.000000000000000e+00, -2.272130367780883e+01, -9.505623306003102e+01, 0.000000000000000e+00, -7.163765802373761e+01, -6.343961533769116e+02, 0.000000000000000e+00, -6.259859673117560e+02, -3.775219415684578e+02, 0.000000000000000e+00, -2.315502546542679e+02, -3.344031167867958e+01, 0.000000000000000e+00, -2.940676032931227e+01, -1.551305082525775e+03, 0.000000000000000e+00, -1.616168117152989e+02, -1.551305082524468e+03, 0.000000000000000e+00, -1.616168117159228e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05