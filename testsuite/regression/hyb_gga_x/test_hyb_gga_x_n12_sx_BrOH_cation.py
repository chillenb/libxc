
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_x_n12_sx_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.564295199649660e+01, -1.564288468025105e+01, -1.564277677357906e+01, -1.564377414202239e+01, -1.564322441084235e+01, -1.564322441084235e+01, -2.699895671038558e+00, -2.699817805696111e+00, -2.698076321859304e+00, -2.701611395538735e+00, -2.699867690350066e+00, -2.699867690350066e+00, -5.439889851708779e-01, -5.442943361309270e-01, -5.526336790295252e-01, -5.559513963290545e-01, -5.553390527151307e-01, -5.553390527151307e-01, -2.093831555455935e-01, -2.087183929611343e-01, -6.251529971163392e-01, -1.851757529776648e-01, -2.059079939878562e-01, -2.059079939878561e-01, -2.644230925341819e-02, -2.768663067643611e-02, -9.213898027243458e-02, -1.594275079118709e-02, -2.167501791693979e-02, -2.167501791693980e-02, -3.562439114436978e+00, -3.556736595893850e+00, -3.562209683123555e+00, -3.557175471509502e+00, -3.559534647726262e+00, -3.559534647726262e+00, -1.573160056257852e+00, -1.586910282447675e+00, -1.557808720031122e+00, -1.570266640507924e+00, -1.587672871609815e+00, -1.587672871609815e+00, -4.352781888703479e-01, -4.214485745716920e-01, -4.092021890375567e-01, -3.817147836204721e-01, -4.369992060056594e-01, -4.369992060056596e-01, -1.390560087306447e-01, -2.308978056658002e-01, -1.323189160922498e-01, -1.180754911831315e+00, -1.584290003904321e-01, -1.584290003904321e-01, -1.247678513546826e-02, -1.560515896229028e-02, -1.210752656858139e-02, -1.096330965399431e-01, -1.502342300075301e-02, -1.502342300075302e-02, -3.763298958955873e-01, -3.875991243537071e-01, -3.837776930481752e-01, -3.804965451727840e-01, -3.821497622652894e-01, -3.821497622652894e-01, -3.626495417856371e-01, -3.982407771013994e-01, -3.922036833023887e-01, -3.829476499014287e-01, -3.878282727940573e-01, -3.878282727940573e-01, -4.435905163420800e-01, -2.659218861919709e-01, -2.803709155113095e-01, -2.906460896803506e-01, -2.830459372531804e-01, -2.830459372531804e-01, -3.685208851213270e-01, -9.049134997673725e-02, -1.018261938065903e-01, -2.640063642715845e-01, -1.223035694273824e-01, -1.223035694273825e-01, -3.569372745033279e-02, -4.352758363264066e-03, -8.997852231546269e-03, -1.183334852862153e-01, -1.386299301534150e-02, -1.386299301534149e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_x_n12_sx_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.088912870432338e+01, -1.088909278920462e+01, -1.088888027753378e+01, -1.088891247972277e+01, -1.088846942676352e+01, -1.088821630891938e+01, -1.089126574777750e+01, -1.089178370132658e+01, -1.088896115125224e+01, -1.089063978215209e+01, -1.088896115125224e+01, -1.089063978215209e+01, -2.516798236153597e+00, -2.514493359136034e+00, -2.516521089019910e+00, -2.514067462824410e+00, -2.508311006916091e+00, -2.506178791091358e+00, -2.522958917149690e+00, -2.520484323094230e+00, -2.517291219582208e+00, -2.513278462305843e+00, -2.517291219582208e+00, -2.513278462305843e+00, -5.288417516756667e-01, -5.287620318828040e-01, -5.286745104500147e-01, -5.286578076599127e-01, -5.219550864047664e-01, -5.175524728560131e-01, -5.250642864951409e-01, -5.263126161324649e-01, -5.293263612182316e-01, -4.750930989446890e-01, -5.293263612182316e-01, -4.750930989446890e-01, -1.422151092834933e-01, -1.348594588319230e-01, -1.364857133209654e-01, -1.295269672588900e-01, -6.252887523597309e-01, -6.272201999793093e-01, -2.052424264922531e-01, -2.055655247689970e-01, -1.442989999720442e-01, -1.935731946116029e-01, -1.442989999720441e-01, -1.935731946116029e-01, -3.323193871116077e-02, -3.501412770925825e-02, -3.459056601699347e-02, -3.667841783067026e-02, -1.000238330069006e-01, -1.016634510664858e-01, -2.107186143286424e-02, -2.074851003752197e-02, -3.008212085611286e-02, -1.799420413851573e-02, -3.008212085611286e-02, -1.799420413851578e-02, -2.626344552496328e+00, -2.626035931187240e+00, -2.621654251374635e+00, -2.621497440097872e+00, -2.626105727929450e+00, -2.625895946366115e+00, -2.622078803587283e+00, -2.621771181300714e+00, -2.623907693510909e+00, -2.623738906221145e+00, -2.623907693510909e+00, -2.623738906221145e+00, -2.418660026647006e+00, -2.418563375642749e+00, -2.416575903504002e+00, -2.416327997169203e+00, -2.459946793235446e+00, -2.448580028356377e+00, -2.461764315575351e+00, -2.449767273195009e+00, -2.379969117107628e+00, -2.412398937167470e+00, -2.379969117107628e+00, -2.412398937167470e+00, -4.430563151740272e-01, -4.418997816895212e-01, -4.939981848959661e-01, -4.946767166163250e-01, -4.007266135551302e-01, -4.163763802383368e-01, -4.329645823845743e-01, -4.453135456585611e-01, -4.605154549948152e-01, -4.382237116980837e-01, -4.605154549948153e-01, -4.382237116980838e-01, -1.915962010446610e-01, -1.930118121492758e-01, -2.120306700147613e-01, -2.111337927033447e-01, -1.732077786720535e-01, -1.824238550602465e-01, -1.193146486616833e+00, -1.192846385932405e+00, -2.064472577509348e-01, -1.953942855968197e-01, -2.064472577509348e-01, -1.953942855968197e-01, -1.611545981984832e-02, -1.671124218577456e-02, -2.033098407047915e-02, -2.061687909696858e-02, -1.546694681786664e-02, -1.631944914663375e-02, -1.236282458219788e-01, -1.241848951788525e-01, -1.579200269607020e-02, -2.120976883265445e-02, -1.579200269607021e-02, -2.120976883265445e-02, -4.731253149984010e-01, -4.749762341758188e-01, -4.592103885276323e-01, -4.610898751425806e-01, -4.636559356882836e-01, -4.655600161725325e-01, -4.677220780072017e-01, -4.695578961848786e-01, -4.656484197130902e-01, -4.675182621728390e-01, -4.656484197130902e-01, -4.675182621728390e-01, -4.659103446623999e-01, -4.673256626757870e-01, -3.626603377789729e-01, -3.646929758037873e-01, -3.858682971323205e-01, -3.878310483462050e-01, -4.104104575561789e-01, -4.118987449386350e-01, -3.973899213759849e-01, -3.989449381452121e-01, -3.973899213759849e-01, -3.989449381452121e-01, -5.125915006372156e-01, -5.140340630629466e-01, -1.770924413711394e-01, -1.758746781556815e-01, -1.612298916412686e-01, -1.614673309128093e-01, -2.352282540570656e-01, -2.371617215360922e-01, -1.857808752397007e-01, -1.858944682990840e-01, -1.857808752397008e-01, -1.858944682990840e-01, -3.274663311923364e-01, -3.306805528295302e-01, -9.973527548516743e-02, -9.993935052743676e-02, -1.097324787857975e-01, -1.111441362774451e-01, -2.342922801912131e-01, -2.395958559784019e-01, -1.478318083647336e-01, -1.540230771291174e-01, -1.478318083647334e-01, -1.540230771291174e-01, -4.485215925690788e-02, -4.613937641212715e-02, -5.773807295541181e-03, -5.786123106805662e-03, -1.150942512129045e-02, -1.220877212014965e-02, -1.419107501848374e-01, -1.440280222523029e-01, -1.499163204565954e-02, -1.957371524077511e-02, -1.499163204565953e-02, -1.957371524077508e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_x_n12_sx_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.045854977011916e-08, 0.000000000000000e+00, -5.045898902737006e-08, -5.046019132818973e-08, 0.000000000000000e+00, -5.046018023838805e-08, -5.046225836845653e-08, 0.000000000000000e+00, -5.046403370991836e-08, -5.044380042005310e-08, 0.000000000000000e+00, -5.044048129279337e-08, -5.045968963844826e-08, 0.000000000000000e+00, -5.044755431562112e-08, -5.045968963844826e-08, 0.000000000000000e+00, -5.044755431562112e-08, -3.494854659873621e-05, 0.000000000000000e+00, -3.504516626704468e-05, -3.495998637780731e-05, 0.000000000000000e+00, -3.506291185782347e-05, -3.530143174629646e-05, 0.000000000000000e+00, -3.539030211147596e-05, -3.468835326678276e-05, 0.000000000000000e+00, -3.479047492407320e-05, -3.493384880179377e-05, 0.000000000000000e+00, -3.508811406065806e-05, -3.493384880179377e-05, 0.000000000000000e+00, -3.508811406065806e-05, -1.378301199372471e-02, 0.000000000000000e+00, -1.415367870349223e-02, -1.367701049383350e-02, 0.000000000000000e+00, -1.412161567900337e-02, -1.270164049032481e-02, 0.000000000000000e+00, -1.247610683213613e-02, -1.213345898405008e-02, 0.000000000000000e+00, -1.222158099149237e-02, -1.465511101069380e-02, 0.000000000000000e+00, -1.247191378237371e-02, -1.465511101069380e-02, 0.000000000000000e+00, -1.247191378237371e-02, -2.012107893752506e+00, 0.000000000000000e+00, -1.992186675520315e+00, -2.087888205112475e+00, 0.000000000000000e+00, -2.031422298820952e+00, -7.404742350345731e-03, 0.000000000000000e+00, -8.047792746830502e-03, -8.789945689406556e-01, 0.000000000000000e+00, -9.035995693825623e-01, -1.761933550518557e+00, 0.000000000000000e+00, 1.268282761503883e+00, -1.761933550518558e+00, 0.000000000000000e+00, 1.268282761503887e+00, -4.590971534186419e+01, 0.000000000000000e+00, -4.488426804345847e+01, -4.790967306874706e+01, 0.000000000000000e+00, -4.695381200653625e+01, -7.017726299603230e+00, 0.000000000000000e+00, -6.146618412618232e+00, -4.545419296698873e+01, 0.000000000000000e+00, -4.430515592215394e+01, -4.682346215825778e+01, 0.000000000000000e+00, -1.279138705001261e+02, -4.682346215826624e+01, 0.000000000000000e+00, -1.279138704999504e+02, -1.704142753374537e-05, 0.000000000000000e+00, -1.705259653603539e-05, -1.713711022857534e-05, 0.000000000000000e+00, -1.714515650475367e-05, -1.704620217319463e-05, 0.000000000000000e+00, -1.705538977949331e-05, -1.712830991369303e-05, 0.000000000000000e+00, -1.713948135629987e-05, -1.709086328702308e-05, 0.000000000000000e+00, -1.709916199694489e-05, -1.709086328702308e-05, 0.000000000000000e+00, -1.709916199694489e-05, 8.152905387381029e-05, 0.000000000000000e+00, 8.154926046892349e-05, 7.514015889608785e-05, 0.000000000000000e+00, 7.530285353277262e-05, 9.546000550010960e-05, 0.000000000000000e+00, 9.165748894025727e-05, 9.053594578007806e-05, 0.000000000000000e+00, 8.657264504553954e-05, 6.585971670251871e-05, 0.000000000000000e+00, 7.663870384588814e-05, 6.585971670251871e-05, 0.000000000000000e+00, 7.663870384588814e-05, -3.779673510708440e-02, 0.000000000000000e+00, -3.824201007326722e-02, -4.186451133068386e-02, 0.000000000000000e+00, -4.200224591079667e-02, -5.121607302951816e-02, 0.000000000000000e+00, -4.620942604788342e-02, -6.411856184607576e-02, 0.000000000000000e+00, -5.686586055863373e-02, -3.465613097704134e-02, 0.000000000000000e+00, -4.127787389030489e-02, -3.465613097704134e-02, 0.000000000000000e+00, -4.127787389030491e-02, 1.858179976873617e+00, 0.000000000000000e+00, 1.803717195071180e+00, -8.330764414218296e-01, 0.000000000000000e+00, -8.377986874525695e-01, 2.110670064309352e+00, 0.000000000000000e+00, 1.987164760881333e+00, -9.999274723832365e-04, 0.000000000000000e+00, -1.001919712443210e-03, 8.160446271412455e-01, 0.000000000000000e+00, -4.138079622712407e-01, 8.160446271412455e-01, 0.000000000000000e+00, -4.138079622712407e-01, -6.564358588826487e+01, 0.000000000000000e+00, -5.662977091945692e+01, -5.518031398898497e+01, 0.000000000000000e+00, -5.086308422329201e+01, -3.229364638574431e+02, 0.000000000000000e+00, -3.574891211406193e+02, 2.815105261157474e-01, 0.000000000000000e+00, 3.998228667954727e-01, -1.603509653023876e+02, 0.000000000000000e+00, -1.526336517401949e+02, -1.603509653023910e+02, 0.000000000000000e+00, -1.526336517401704e+02, -6.084669827766626e-02, 0.000000000000000e+00, -6.016715748695747e-02, -5.648836807773362e-02, 0.000000000000000e+00, -5.589949800349947e-02, -5.796801253671656e-02, 0.000000000000000e+00, -5.735865618628396e-02, -5.924745136299614e-02, 0.000000000000000e+00, -5.859267849656068e-02, -5.860381335170357e-02, 0.000000000000000e+00, -5.797133673228212e-02, -5.860381335170357e-02, 0.000000000000000e+00, -5.797133673228212e-02, -6.848388280154871e-02, 0.000000000000000e+00, -6.775649562116801e-02, -5.772871332769405e-02, 0.000000000000000e+00, -5.700574699060570e-02, -5.802612353572228e-02, 0.000000000000000e+00, -5.745845342727192e-02, -6.145717484065800e-02, 0.000000000000000e+00, -6.082852913352136e-02, -5.961418857827359e-02, 0.000000000000000e+00, -5.894341016272637e-02, -5.961418857827359e-02, 0.000000000000000e+00, -5.894341016272637e-02, -3.510039522686111e-02, 0.000000000000000e+00, -3.517984060397775e-02, -7.326752726647108e-01, 0.000000000000000e+00, -7.296253921445517e-01, -5.469016833942240e-01, 0.000000000000000e+00, -5.377069444829636e-01, -2.524941244904888e-01, 0.000000000000000e+00, -2.474004116621191e-01, -3.842877880833176e-01, 0.000000000000000e+00, -3.862395254813866e-01, -3.842877880833177e-01, 0.000000000000000e+00, -3.862395254813867e-01, -8.245875068247246e-02, 0.000000000000000e+00, -8.087474101512014e-02, -6.692767878781081e+00, 0.000000000000000e+00, -6.596112389322347e+00, -2.233577608142307e+00, 0.000000000000000e+00, -1.841442860893231e+00, -3.322475268638105e-01, 0.000000000000000e+00, -3.152160518385379e-01, 1.836335586217614e+00, 0.000000000000000e+00, 1.609864821356387e+00, 1.836335586217517e+00, 0.000000000000000e+00, 1.609864821356358e+00, -3.333282400523885e+01, 0.000000000000000e+00, -3.365968408267396e+01, -2.133018138617640e+02, 0.000000000000000e+00, -3.777958133670826e+02, -1.278075409158612e+02, 0.000000000000000e+00, -1.355500887463766e+02, 1.546314662752549e+00, 0.000000000000000e+00, 1.641523784568594e+00, -3.319706025605494e+02, 0.000000000000000e+00, -1.599966400219908e+02, -3.319706025605186e+02, 0.000000000000000e+00, -1.599966400220267e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05